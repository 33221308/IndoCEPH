# utils/angle_config.py
# Index sesuai YAML:
# names: ['ANS','Gna','Go','InterM','LA','LI','Me','Nasion','Orb','PNS','Po','Pog','Point A','Point B','Sella','UA','UI']

ANS, Gna, Go, InterM, LA, LI, Me, Nasion, Orb, PNS, Po, Pog, A, B, S, UA, UI = range(17)

# 3 sudut langsung (3 titik, sudut di tengah)
TRUE_ANGLES = [
    ("SNA", (S, Nasion, A)),
    ("SNB", (S, Nasion, B)),
    ("ANB", (A, Nasion, B)),
]

# 5 sudut antar-garis (2 garis)
LINE_ANGLES = [
    ("SN_Mand",      ((S, Nasion), (Go, Me))),        # SN vs mandibular plane
    ("SN_Occlusal",  ((S, Nasion), (UI, InterM))),    # SN vs occlusal (UI–InterM)
    ("Pal_Occlusal", ((ANS, PNS),  (UI, InterM))),    # palatal vs occlusal
    ("FH_Mand",      ((Po, Orb),   (Go, Me))),        # FH vs mandibular
    ("Interincisal", ((UI, UA),    (LI, LA))),        # UI axis vs LI axis
]

# Bobot (bisa kamu ubah sewaktu-waktu)
ANGLE_WEIGHTS = {
    "SNA": 1.0, "SNB": 1.0, "ANB": 1.0,
    "SN_Mand": 1.0, "SN_Occlusal": 1.0,
    "Pal_Occlusal": 1.0, "FH_Mand": 1.0,
    "Interincisal": 1.0
}
