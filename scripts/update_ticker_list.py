#!/usr/bin/env python3
"""
Update ticker list with additional IBOV and SMLL constituents.
"""

import json
from pathlib import Path
from datetime import datetime

# Current tickers
data_path = Path(__file__).parent.parent / 'data' / 'validated_tickers.json'

with open(data_path, 'r') as f:
    data = json.load(f)

current = set(data['all_tickers'])
ibov_current = set(data['indices']['IBOV_TOP50'])
smll_current = set(data['indices']['SMLL_LIQUID'])

# IBOV stocks to add (confirmed constituents)
additional_ibov = [
    "BBDC3.SA", "BPAC11.SA", "CMIG3.SA", "CPLE6.SA", "CSNA3.SA",
    "EMBR3.SA", "ELET3.SA", "ELET6.SA", "ENBR3.SA", "CIEL3.SA",
    "GOAU4.SA", "JBSS3.SA", "MRFG3.SA", "NTCO3.SA", "PRIO3.SA",
    "SMTO3.SA", "SLCE3.SA", "TIMS3.SA", "RRRP3.SA", "ALSO3.SA",
    "ARZZ3.SA", "AZUL4.SA", "BRAP4.SA", "BRFS3.SA", "BRKM5.SA",
    "BBSE3.SA", "CRFB3.SA", "CCRO3.SA", "CMIN3.SA", "CVCB3.SA",
    "GOLL4.SA", "IGTI11.SA", "PETR3.SA", "SOMA3.SA",
]

# SMLL stocks to add (liquid small caps)
additional_smll = [
    "ALPA3.SA", "BALM3.SA", "BMOB3.SA", "BSEV3.SA", "CBAV3.SA",
    "CAML3.SA", "CESP3.SA", "CESP5.SA", "CESP6.SA", "COCE3.SA",
    "COCE5.SA", "COCE6.SA", "CPRE3.SA", "CTNM3.SA", "DTCY3.SA",
    "ELMD3.SA", "ENAT3.SA", "ETER3.SA", "FIBR3.SA", "FRIO3.SA",
    "GEPA3.SA", "GEPA4.SA", "GPAR3.SA", "HBRE3.SA", "INEP3.SA",
    "INEP4.SA", "JALL3.SA", "KEPL3.SA", "LCAM3.SA", "LPAN3.SA",
    "LUPA3.SA", "MATD3.SA", "MDNE3.SA", "MELK3.SA", "MILS3.SA",
    "MMXM3.SA", "MYPK3.SA", "NEOE3.SA", "NGRD3.SA", "ODER3.SA",
    "OFER3.SA", "OPCT3.SA", "ORVR3.SA", "PARD3.SA", "PATS3.SA",
    "PDGR3.SA", "PGMN3.SA", "PKSA3.SA", "PLAS3.SA", "PMAM3.SA",
    "PNVL3.SA", "PNVL4.SA", "POWE3.SA", "PTCA3.SA", "PTCA4.SA",
    "PTNT3.SA", "QGEC3.SA", "QGEC4.SA", "RCSL3.SA", "RCSL4.SA",
    "RNEW11.SA", "RNEW3.SA", "RNEW4.SA", "ROMI3.SA", "RPAD3.SA",
    "RPAD5.SA", "RSID3.SA", "SAPR11.SA", "SAPR3.SA", "SAPR4.SA",
    "SEDU3.SA", "SEIV3.SA", "SEIV4.SA", "SGPS3.SA", "SHOW3.SA",
    "SIMH3.SA", "SMFT3.SA", "SOND3.SA", "SOND5.SA", "SOND6.SA",
    "SPRI3.SA", "STBP3.SA", "STBP11.SA", "SULA11.SA", "TASA3.SA",
    "TASA4.SA", "TCIN3.SA", "TECN3.SA", "TGMA3.SA", "TLEF3.SA",
    "TLEF4.SA", "TMCP3.SA", "TMCP4.SA", "TNLP3.SA", "TNLP4.SA",
    "TNEP3.SA", "TNEP4.SA", "TOYR3.SA", "TPIS3.SA", "TRAD3.SA",
    "TRIS3.SA", "TRPN3.SA", "TRUS3.SA", "UCAS3.SA", "UNIP3.SA",
    "UNIP5.SA", "UNIP6.SA", "USIM3.SA", "USIM6.SA", "VIVA3.SA",
    "VIVR3.SA", "VULC3.SA", "WIZC3.SA", "WIZS3.SA", "ZAMP3.SA",
]

# Filter out already existing
new_ibov = [t for t in additional_ibov if t not in current]
new_smll = [t for t in additional_smll if t not in current]

print(f"Current: {len(current)} stocks")
print(f"Adding IBOV: {len(new_ibov)}")
print(f"Adding SMLL: {len(new_smll)}")

# Update sets
all_tickers = sorted(list(current | set(new_ibov) | set(new_smll)))
ibov_updated = sorted(list(ibov_current | set(new_ibov)))
smll_updated = sorted(list(smll_current | set(new_smll)))

# Create updated data
updated_data = {
    'description': 'Brazilian stock tickers for trading analysis (IBOV + SMLL)',
    'generated_at': datetime.now().isoformat(),
    'total_tickers': len(all_tickers),
    'indices': {
        'IBOV': ibov_updated,
        'SMLL_LIQUID': smll_updated,
    },
    'all_tickers': all_tickers,
}

# Backup
backup_path = data_path.with_suffix('.json.backup-' + datetime.now().strftime('%Y%m%d'))
with open(backup_path, 'w') as f:
    json.dump(data, f, indent=2)

# Save updated
with open(data_path, 'w') as f:
    json.dump(updated_data, f, indent=2)

print(f"\nUpdated: {len(all_tickers)} total stocks")
print(f"  IBOV: {len(ibov_updated)}")
print(f"  SMLL: {len(smll_updated)}")
print(f"Backup: {backup_path}")