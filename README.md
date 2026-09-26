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
```

For the paper's HS300 evaluation window, run with full history (2008 warm-up; train 2009–2019, validation 2019, test 2020–2024):

```powershell
python raven_tushare_reproduction.py --mode all
```

The 2019 calendar year is the held-out validation subset and is excluded from gradient fitting and target scaling. The test window follows the paper. Tushare access, historical membership permissions, and local compute determine whether the full run is feasible.

See [RAVEN_Tushare_模块讲解.md](RAVEN_Tushare_模块讲解.md) for methodology, limitations, and outputs.
