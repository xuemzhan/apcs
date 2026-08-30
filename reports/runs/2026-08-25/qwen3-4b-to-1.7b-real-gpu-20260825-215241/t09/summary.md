# T09 Main Capability

> ⚠️ **offline demo**：T09 得分为合成数据，不可作为真实能力迁移的证据；design.md §75 要求真实 Test 上 CHG>0 才支持 Runtime Capability Transfer。

- Student: 0.4933
- Teacher: 0.7953
- Teacher gap: 0.3020

## Per-method (§37)
| method | score | std | retention | CHG | TGRR | JCR vs Student |
| ------ | ----: | --: | --------: | --: | ---: | -------------: |
| text | 0.5505 | 0.0264 | 1.1160 | +0.0572 | +0.1894 | 0.5417 |
| ridge | 0.5776 | 0.0250 | 1.1708 | +0.0842 | +0.2789 | 0.4375 |
| base_only | 0.6111 | 0.0242 | 1.2388 | +0.1178 | +0.3901 | 0.4479 |
| base_plus_adv | 0.6559 | 0.0252 | 1.3296 | +0.1626 | +0.5384 | 0.5000 |
| full_apcs | 0.7048 | 0.0287 | 1.4288 | +0.2115 | +0.7003 | 0.5729 |

## Bootstrap CI on CHG (base+adv − student, n=32)
- point=+0.1626, 95% CI=[+0.1493, +0.1769]

## Paired Permutation Test (§51, n_perm=2000)
- p-value = 0.0005

## Gap Strata (§48)
| strata | n | gap | CHG | TGRR |
| ------ | -: | --: | --: | ---: |
| low | 11 | 0.2594 | +0.1480 | +0.5707 |
| medium | 10 | 0.3047 | +0.1511 | +0.4961 |
| high | 11 | 0.3423 | +0.1876 | +0.5483 |
