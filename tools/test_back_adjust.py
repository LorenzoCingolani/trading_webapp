"""Check back_adjust() against the worked example in Carver's Advanced Futures Trading
Strategies, Appendix B (page 602): roll from contract C to D on 16 January 2022.

Run from the project root:  python tools/test_back_adjust.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from merge_near_far import back_adjust  # noqa: E402

# Book's back-adjusted series before the roll (2-16 Jan), then contract D after it.
# 17 Jan is not in the book: it's the first day on D, as Bloomberg's generic 1st contract would show it.
dates = ['02/01', '03/01', '06/01', '07/01', '08/01', '09/01', '10/01', '13/01', '14/01', '15/01', '16/01', '17/01']
near = [100.8, 101.0, 101.1, 100.7, 100.4, 99.2, 99.5, 99.9, 100.1, 101.6, 101.0, 101.7]
far = [None] * 10 + [101.5, None]          # D's price on the roll date
switches = [11]                            # 17 Jan = first day on D

# Book: "We take the difference ... 101.5 - 101.0 = 0.5. We now add this difference to the
# entire back-adjusted price series". The book prints 100.5 for 14 Jan (100.1 + 0.5), a typo.
expected = [101.3, 101.5, 101.6, 101.2, 100.9, 99.7, 100.0, 100.4, 100.6, 102.1, 101.5, 101.7]

adjusted, gap, _ = back_adjust(pd.Series(near), pd.Series(far, dtype=float), switches)

ok = True
for d, a, e in zip(dates, adjusted, expected):
    match = abs(a - e) < 1e-9
    ok &= match
    print(f'{d}  adjusted {a:7.2f}  book {e:7.2f}  {"OK" if match else "MISMATCH"}')
print(f'gap on roll date: {gap[10]:.2f} (book: 0.5)')
assert ok and abs(gap[10] - 0.5) < 1e-9, 'back_adjust does not match the book'

# Early roll (Carver's calendar), simple example: contract A, then B. We roll on day 2 (gap
# B - A = 104 - 101 = 3) while Bloomberg's 1st contract only switches to B on day 4. Between,
# the held contract is B, so its prices come from far.
near2 = pd.Series([100.0, 101.0, 102.0, 106.0, 104.0])   # A, A, A, then B (Bloomberg switch)
far2 = pd.Series([103.0, 104.0, 105.0, 107.0, 105.0])    # B, B, B, then next contract
adj2, gap2, cur2 = back_adjust(near2, far2, [(1, 3)])
print('early roll adjusted', list(adj2), 'held', list(cur2))
assert list(adj2) == [103.0, 104.0, 105.0, 106.0, 104.0]   # B's prices from day 2 onwards
assert list(cur2) == [100.0, 104.0, 105.0, 106.0, 104.0]
print('PASS')
