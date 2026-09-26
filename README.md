# RAVEN reproduction (Tushare / HS300)

This repository contains an independent, auditable implementation of the method in [RAVEN (arXiv:2606.24062v1)](https://arxiv.org/abs/2606.24062), using Tushare daily data. The paper's full private factor list and Qlib pipeline are not public, so this is a method-level reproduction rather than a bit-for-bit reproduction of the authors' data pipeline.

## Security first

A Tushare token was previously committed in the Python source. Treat that credential as compromised and rotate/revoke it in Tushare before running this code. The script now reads `TUSHARE_TOKEN` from the environment and does not include a credential. Never commit the token.

PowerShell:

```powershell
$env:TUSHARE_TOKEN = "your-new-token"
```

Install dependencies with `pip install -r requirements.txt`, then run a small end-to-end experiment:

```powershell
python raven_tushare_reproduction.py --mode all --max-stocks 20 --epochs 3
