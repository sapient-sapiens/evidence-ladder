SPECIES_CODES = {
    "ALPHA_DRACONIAN",
    "ANDROMEDAN",
    "AQUARIAN_MANTIS",
    "ARCTURIAN",
    "CENTAURI_SYNTH",
    "JOVIAN_GASFORM",
    "KAIJU_MICRO",
    "LUNA_SECURID",
    "ORION_GRAYS",
    "SIRIUS_AVIAN",
    "TRIANGULAN",
    "VENUSIAN_MYCELIAL",
}

HOME_WORLDS = {
    "Barnard-c",
    "Eris Relay",
    "Europa Station",
    "Gliese-581g",
    "Kepler-186f",
    "Luyten-b",
    "Mars Dome-7",
    "Proxima-b",
    "Sirius Outpost",
    "TRAPPIST-1e",
    "Titan Freeport",
    "Wolf-1061c",
    "Zeta Reticuli",
}

VISA_CLASSES = {"XW-1", "XW-2", "DIP-1", "MED-3", "TRANSIT-7"}

PURPOSES = {
    "archive audit",
    "cultural exchange",
    "diplomatic",
    "field repair",
    "medical consult",
    "reactor maintenance",
    "research",
    "transit",
    "translation",
    "xenobotany",
}

FEE_STATUSES = {"paid", "waived", "unpaid", "unknown"}

RISK_FLAG_ATOMS = {
    "active_warrant",
    "biohazard_red",
    "identity_conflict",
    "illegible_biometrics",
    "memory_tampering",
    "planetary_embargo",
    "rescinded_denial",
    "sponsor_mismatch",
}

DISQUALIFYING_FLAGS = {
    "memory_tampering",
    "planetary_embargo",
    "active_warrant",
    "biohazard_red",
}

REVIEW_ONLY_FLAGS = {
    "identity_conflict",
    "sponsor_mismatch",
    "illegible_biometrics",
    "rescinded_denial",
}

REVOKED_SPONSORS = {"SPN-0007", "SPN-0139", "SPN-4040"}

# Publicly documented revoked sponsors are also unsafe approval evidence for
# diplomatic packets, where the hard-deny rule is waived.
SUSPICIOUS_SPONSORS = REVOKED_SPONSORS

# Public data cut date (zip version 2026-07-07). Stale = arrival >180 days before this,
# except DIP-1. On train this matches DENIED labels with zero false APPROVED hits.
PACKET_RECEIPT_DATE = "2026-07-07"
STALE_DAYS = 180

CRITICAL_FIELDS = (
    "applicant_name",
    "species_code",
    "home_world",
    "visa_class",
    "sponsor_id",
    "arrival_date",
    "declared_purpose",
    "fee_status",
)
