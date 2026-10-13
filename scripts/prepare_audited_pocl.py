"""Separate correctness variant, NOT a compatibility-only reproduction."""
from pathlib import Path
import difflib
root=Path(__file__).resolve().parents[1];src=root/'upstream/POCL';dst=root/'work/pocl-audited';dst.mkdir(exist_ok=True)
patch=[]
for name in ['main.py','models.py','tool.py']:
 original=(src/name).read_text();s=original
 if name=='models.py':
  s=s.replace('F.dropout(x,p=0.5)','F.dropout(x,p=0.5,training=self.training)')
  s=s.replace('overall_importance[n] += m * overall_importance[n]', 'overall_importance[n] = m * overall_importance[n]')
 if name=='main.py':
  # Preprocessor writes [normal,fraud]. Align pretrain target labels with task loss.
  s=s.replace('contrast_dataset[i * 2 + 1]', 'contrast_dataset[__FRAUD__]').replace('contrast_dataset[i * 2]', 'contrast_dataset[i * 2 + 1]').replace('contrast_dataset[__FRAUD__]', 'contrast_dataset[i * 2]')
  s=s.replace('        print("Offline Epoch:", j, "Loss:", loss_sum)','        print("Offline Epoch:", j, "Loss:", loss_sum)\n    return pretrain_data[:pretrain_date]')
 (dst/name).write_text(s)
 patch.extend(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='upstream/'+name,tofile='audited/'+name))
(root/'scripts/audited-corrections.patch').write_text(''.join(patch))
