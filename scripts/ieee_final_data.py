"""Chronology-checked, bounded raw-partition access and transform-only features."""
import csv
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from prepare_ieee_windows import CATEGORICAL, NUMERIC, PREDICTORS, METADATA

PARTITIONS = ('development', 'validation', 'reserved_evaluation')


def validate_preparation(prepared):
    start = prepared['time_span']['min_transaction_dt']
    end = prepared['time_span']['max_transaction_dt']
    if not end > start or not all(prepared['partition_checks'].values()):
        raise ValueError('Invalid chronological preparation')
    parts = prepared['partitions']
    previous = start-1
    for name in PARTITIONS:
        part = parts[name]
        if (part['row_count'] <= 0 or part['min_transaction_dt'] <= previous
                or part['max_transaction_dt'] < part['min_transaction_dt']):
            raise ValueError('Partitions overlap or are empty')
        previous = part['max_transaction_dt']
    if (parts['development']['min_transaction_dt'] != start or previous != end
            or sum(p['row_count'] for p in parts.values()) != prepared['source_row_count']):
        raise ValueError('Preparation coverage mismatch')
    for name in PARTITIONS:
        for dt in (parts[name]['min_transaction_dt'], parts[name]['max_transaction_dt']):
            if partition_at(dt, start, end) != name:
                raise ValueError('Range disagrees with rational partition boundary')


def partition_at(dt, start, end):
    offset = 5*(dt-start)
    if not start <= dt <= end:
        raise ValueError('Timestamp outside source range')
    return 'development' if offset < 3*(end-start) else 'validation' if offset < 4*(end-start) else 'reserved_evaluation'


def load_partition(source, prepared, phase, expected_prefix=None):
    """Read only the authorized logical CSV prefix; never scan beyond its end.

    This function is called only by explicit future validation/reserved commands.
    Validation cannot hash the full shared CSV without reading reserved records.
    """
    if phase not in ('validation', 'reserved'):
        raise ValueError('Raw access requires validation or reserved phase')
    validate_preparation(prepared)
    source = Path(source)
    if source.stat().st_size != prepared['source_bytes']:
        raise ValueError('Source size changed')
    parts = prepared['partitions']
    prefix_count = parts['development']['row_count']+parts['validation']['row_count']
    target = 'validation' if phase == 'validation' else 'reserved_evaluation'
    offset = parts['development']['row_count'] if phase == 'validation' else prefix_count
    stop = offset+parts[target]['row_count']
    digest = hashlib.sha256()
    prefix_hash = None
    columns = {name: [] for name in PREDICTORS+METADATA}
    ids = set()
    previous_dt = None
    observed = {name: {'count': 0, 'min': None, 'max': None} for name in PARTITIONS}
    start, end = (prepared['time_span'][key] for key in ('min_transaction_dt', 'max_transaction_dt'))
    with source.open('rb') as stream:
        def lines():
            while True:
                raw = stream.readline()
                if not raw:
                    return
                digest.update(raw)
                yield raw.decode('utf-8')
        reader = csv.reader(lines(), strict=True)
        header = next(reader)
        if len(set(header)) != len(header) or not set(columns).issubset(header):
            raise ValueError('Missing or repeated CSV columns')
        positions = {name: header.index(name) for name in columns}
        for index in range(stop):
            try:
                row = next(reader)
            except StopIteration as error:
                raise ValueError('Source ended before declared partition count') from error
            if len(row) != len(header):
                raise ValueError('Wrong source column count')
            transaction_id = int(row[positions['TransactionID']])
            dt = int(row[positions['TransactionDT']])
            name = partition_at(dt, start, end)
            expected = 'development' if index < parts['development']['row_count'] else 'validation' if index < prefix_count else 'reserved_evaluation'
            if name != expected or (previous_dt is not None and dt < previous_dt) or transaction_id in ids:
                raise ValueError('Chronology, partition membership or ID uniqueness failed')
            ids.add(transaction_id); previous_dt = dt
            stats = observed[name]
            stats['count'] += 1
            stats['min'] = dt if stats['min'] is None else stats['min']
            stats['max'] = dt
            if index >= offset:
                for key in columns:
                    value = row[positions[key]]
                    columns[key].append(value if key not in CATEGORICAL or value else '__MISSING__')
            if index+1 == prefix_count:
                prefix_hash = digest.hexdigest()
                if expected_prefix is not None and prefix_hash != expected_prefix:
                    raise ValueError('Validation source prefix changed')
        if phase == 'reserved':
            # EOF inspection is allowed only after explicit reserved access.
            if stream.read(1):
                raise ValueError('Extra rows/bytes beyond preparation count')
            if digest.hexdigest() != prepared['source_sha256']:
                raise ValueError('Complete raw source SHA mismatch')
    for name in PARTITIONS[:2 if phase == 'validation' else 3]:
        stat, part = observed[name], parts[name]
        if (stat['count'], stat['min'], stat['max']) != (part['row_count'], part['min_transaction_dt'], part['max_transaction_dt']):
            raise ValueError('Observed partition range/count changed')
    frame = pd.DataFrame(columns)
    for name in ('TransactionID', 'TransactionDT', 'isFraud'):
        values = pd.to_numeric(frame[name], errors='raise')
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f'Invalid integer metadata: {name}')
        frame[name] = values.astype(np.int64)
    frame['TransactionAmt'] = pd.to_numeric(frame['TransactionAmt'], errors='raise')
    if (not frame.isFraud.isin([0, 1]).all() or not np.isfinite(frame.TransactionAmt).all()
            or (frame.TransactionAmt < 0).any()):
        raise ValueError('Invalid scoring outcomes')
    return frame, {'phase': phase, 'rows': len(frame), 'consumed_rows': stop,
                   'validation_prefix_sha256': prefix_hash,
                   'consumed_source_sha256': digest.hexdigest(),
                   'full_source_verified': phase == 'reserved'}


def transform_frame(frame, preprocessor, feature_names):
    """No fit calls; metadata cannot enter the classifier matrix."""
    if (list(preprocessor.feature_names_in_) != PREDICTORS
            or list(preprocessor.get_feature_names_out()) != feature_names
            or any(any(key in name for key in METADATA) for name in feature_names)):
        raise ValueError('Frozen feature schema changed or contains metadata')
    features = frame[PREDICTORS].copy()
    for name in NUMERIC:
        features[name] = pd.to_numeric(features[name], errors='coerce')
    matrix = sparse.csr_matrix(preprocessor.transform(features), dtype=np.float32)
    if matrix.shape != (len(frame), len(feature_names)) or not np.isfinite(matrix.data).all():
        raise ValueError('Invalid transformed matrix')
    return matrix


def load_preprocessor(path, feature_names):
    preprocessor = joblib.load(path)
    if list(preprocessor.get_feature_names_out()) != feature_names:
        raise ValueError('Preprocessor and feature names disagree')
    return preprocessor
