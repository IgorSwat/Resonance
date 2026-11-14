"""
Words ending in a period that do not end a sentence, per language.

Kept apart rather than merged: Polish "ok." would otherwise stop "That's ok." from ending an English sentence.
"""

ABBREVIATIONS = {
    "en": {"mr.", "mrs.", "ms.", "dr.", "st.", "vs.", "etc.", "e.g.", "i.e.", "prof.", "jr.", "sr."},
    "pl": {"np.", "itd.", "itp.", "tzn.", "tj.", "tzw.", "m.in.", "ul.", "al.", "dr.", "prof.",
           "mgr.", "inż.", "hab.", "płk.", "gen.", "ks.", "św.", "nr.", "pkt.", "str.", "godz.", "tys.",
           "mln.", "mld.", "zł.", "wg.", "ds.", "cdn.", "por.", "zob.", "tel.", "jw."},
    "es": {"sr.", "sra.", "srta.", "sres.", "dr.", "dra.", "ud.", "uds.", "vd.", "vds.", "d.", "dña.",
           "pág.", "págs.", "núm.", "art.", "cap.", "etc.", "ej.", "p.ej.", "aprox.", "lic.", "ing.",
           "prof.", "av.", "avda.", "fig.", "sto.", "sta.", "dto.", "tel.", "vol."},
    "pt": {"sr.", "sra.", "srta.", "dr.", "dra.", "prof.", "profa.", "eng.", "av.", "pág.", "págs.", "núm.",
           "art.", "cap.", "etc.", "ex.", "aprox.", "fig.", "tel.", "vol.", "sto.", "sta.", "lda.", "cia."},
    "de": {"z.b.", "bzw.", "usw.", "d.h.", "u.a.", "ca.", "dr.", "prof.", "nr.", "str.", "vgl.", "evtl.", "ggf.",
           "inkl.", "zzgl.", "bspw.", "etc.", "hr.", "fr.", "jh.", "mio.", "mrd.", "tsd.", "min.", "std.", "abs.",
           "o.ä.", "u.ä.", "z.t.", "s.", "bzgl.", "allg.", "max.", "mind."},
    "it": {"sig.", "sig.ra", "sigg.", "dott.", "dott.ssa", "prof.", "prof.ssa", "ing.", "avv.", "arch.", "geom.",
           "ecc.", "es.", "pag.", "pagg.", "art.", "n.", "nr.", "cap.", "vol.", "tel.", "ca.", "p.es.", "c.a.",
           "s.p.a.", "s.r.l.", "fig.", "cfr.", "rag.", "on.", "mons.", "gen.", "col.", "magg."},
    "fr": {"m.", "mm.", "mme.", "mlle.", "dr.", "pr.", "me.", "mgr.", "st.", "ste.", "etc.", "cf.", "p.ex.", "ex.",
           "env.", "chap.", "art.", "fig.", "vol.", "p.", "pp.", "av.", "bd.", "tél.", "n°.", "c.-à-d.", "c.à.d.",
           "apr.", "av.-j.-c.", "j.-c.", "max.", "min.", "approx."},
}
# Languages whose ordinals are a number with a dot (am 3. Oktober): such a word never ends a sentence.
ORDINAL_DOT = {"de"}
