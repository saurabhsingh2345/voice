# Seed sweep — is a missing grapheme systematic, or one bad sample?

`uv run python -m voiceagent.eval.devanagari --seeds 4`

Chatterbox samples, and the spread at this quality level is wide. A grapheme
that vanishes on one seed proves nothing; one that vanishes on every seed is
the model. Read the survival counts, not the overlaps.

## `m1` [mr] — does `ळ` survive?

Target: सकाळी मी शाळेत लवकर पोहोचलो.

| Seed | Overlap | Grapheme | Heard |
| --- | --- | --- | --- |
| 0 | 0.67 | **no** | सकाई मी शायद लव कर पहुचलो। |
| 1 | 0.71 | **no** | सकाय मी शायद लवकर पहुच लो. |
| 2 | 0.67 | **no** | जकाई मी शायद लव कर पहुचलो. |
| 3 | 0.62 | **no** | सकाई में शायद लब कर पहुच चलो |

**`ळ` survived 0/4 seeds.**

## `m2` [mr] — does `च` survive?

Target: पाच वाजता चहा घेऊया.

| Seed | Overlap | Grapheme | Heard |
| --- | --- | --- | --- |
| 0 | 0.59 | yes | बाच्च बाच्च ताच्छ हां गयुँ या |
| 1 | 0.65 | **no** | आज वाजता जहां घेवया |
| 2 | 0.88 | yes | पाच वाजता चहा घे उया |
| 3 | 0.88 | yes | पाच वाजता चहा घेवया |

**`च` survived 3/4 seeds.**

## `n5` [ne] — does `र्` survive?

Target: यो काम गर्न धेरै समय लाग्छ.

| Seed | Overlap | Grapheme | Heard |
| --- | --- | --- | --- |
| 0 | 0.68 | **no** | युकाम गन धेरे समय लाक्ष |
| 1 | 0.73 | **no** | यो काम गन धेरे समय लाच्च |
| 2 | 0.77 | yes | यो कामगर्ण धेरे समयलाच्च |
| 3 | 0.73 | yes | यो कामकर्ण धेरे समय लाक्ष |

**`र्` survived 2/4 seeds.**

