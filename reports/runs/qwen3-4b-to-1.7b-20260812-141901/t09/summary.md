# T09 Main Capability

> ⚠️ **offline demo**：T09 得分为合成数据，不可作为真实能力迁移的证据；design.md §75 要求真实 Test 上 CHG>0 才支持 Runtime Capability Transfer。

- Student: 0.5029
- Teacher: 0.8069
- Teacher gap: 0.3040

## Per-method (§37)
| method | score | std | retention | CHG | TGRR | JCR vs Student |
| ------ | ----: | --: | --------: | --: | ---: | -------------: |
| text | 0.5537 | 0.0321 | 1.1009 | +0.0508 | +0.1670 | 0.4167 |
| ridge | 0.5899 | 0.0219 | 1.1730 | +0.0870 | +0.2861 | 0.5000 |
| base_only | 0.6057 | 0.0313 | 1.2044 | +0.1028 | +0.3381 | 0.5000 |
| base_plus_adv | 0.6727 | 0.0227 | 1.3377 | +0.1698 | +0.5587 | 0.3333 |
| full_apcs | 0.6767 | 0.0231 | 1.3457 | +0.1738 | +0.5719 | 0.6667 |

## Bootstrap CI on CHG (base+adv − student, n=4)
- point=+0.1698, 95% CI=[+0.1452, +0.2060]

## Paired Permutation Test (§51, n_perm=2000)
- p-value = 0.0580

## Gap Strata (§48)
| strata | n | gap | CHG | TGRR |
| ------ | -: | --: | --: | ---: |
| low | 1 | 0.2858 | +0.1538 | +0.5383 |
| medium | 1 | 0.2901 | +0.1628 | +0.5613 |
| high | 2 | 0.3200 | +0.1814 | +0.5667 |
