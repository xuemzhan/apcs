# Task Report

- TASK_ID: `t12`
- STATUS: **[SIMULATED] OK**
- Generated at: 2026-08-25T23:10:37.177764

> ⚠️ **本任务结果为离线模拟/占位数据（offline demo / placeholder），不可作为真实实验证据**（design.md §75）。

## OBJECTIVE
CKA / principal angle / attn-output cosine / effective rank / head correlation（Figure 7）。

## MODEL_PAIR
- Teacher: `Qwen/Qwen3-4B`
- Student: `Qwen/Qwen3-1.7B`

## DATASET
mmlu

## CONFIG
```json
{
  "context_lengths": [
    512,
    1024
  ],
  "seeds": [
    0,
    1,
    2
  ],
  "mapper": {
    "type": "ridge",
    "rank": 32,
    "separate_kv": true,
    "de_rope": true,
    "de_rope_k": true,
    "de_rope_v": false,
    "ridge_lambda_k": 0.001,
    "ridge_lambda_v": 0.01,
    "real_calibration_samples": 32,
    "real_eval_samples": 16,
    "source_top_k": 2,
    "layer_selection": "proportional",
    "alpha_max": 0.5,
    "rms_calibration": true,
    "shared_basis": false,
    "t06_aggregate_samples": 8
  },
  "advantage": {
    "rank": 16,
    "key": "lowrank",
    "value": "lowrank",
    "source_mixer": true,
    "rms_calibration": true,
    "bounded_alpha": true
  }
}
```

## IMPLEMENTATION
T12 Geometry Diagnostics

## OUTPUT_FILES
- `compliance.json`
- `config.json`
- `geometry.json`
- `metadata.json`
- `metrics.json`
- `provider.json`
- `stdout.log`
- `summary.md`

## KEY_METRICS
```json
{
  "task": "T12",
  "placeholder": true,
  "n_layers_compared": 28,
  "mean_cka": 0.9433895281768621,
  "mean_principal_angle": 2.548617347935253e-08,
  "mean_attn_output_cosine": -0.009333400034569419,
  "mean_head_correlation": 0.06994921342816911,
  "geometry": {
    "placeholder": true,
    "note": "若 placeholder=True，metrics 由合成子空间生成，请用 hidden_states_path 接入真实模型。",
    "per_layer": [
      {
        "student_layer": 0,
        "teacher_layer": 0,
        "cka": 0.9457354688183517,
        "principal_angle": 3.650024149988857e-08,
        "effective_rank_t": 7.9249194812593196,
        "effective_rank_s": 7.939582617814612,
        "attn_output_cosine": -0.019577860246348385,
        "head_correlation": 0.07034124272919326
      },
      {
        "student_layer": 1,
        "teacher_layer": 1,
        "cka": 0.9696686820920613,
        "principal_angle": 3.332000937312528e-08,
        "effective_rank_t": 7.95288602772595,
        "effective_rank_s": 7.946210918506384,
        "attn_output_cosine": -0.04092475106092579,
        "head_correlation": 0.06641749236845898
      },
      {
        "student_layer": 2,
        "teacher_layer": 3,
        "cka": 0.9425522013249913,
        "principal_angle": 2.1073424255447017e-08,
        "effective_rank_t": 7.948473169932046,
        "effective_rank_s": 7.930031466307632,
        "attn_output_cosine": -0.08001103655095992,
        "head_correlation": 0.06397662844620973
      },
      {
        "student_layer": 3,
        "teacher_layer": 4,
        "cka": 0.9535810876286781,
        "principal_angle": 2.5809568279517847e-08,
        "effective_rank_t": 7.9244293624525355,
        "effective_rank_s": 7.939322321934922,
        "attn_output_cosine": -0.07383082697961599,
        "head_correlation": 0.07630843779776068
      },
      {
        "student_layer": 4,
        "teacher_layer": 5,
        "cka": 0.9405847697613884,
        "principal_angle": 2.1073424255447017e-08,
        "effective_rank_t": 7.949678644150717,
        "effective_rank_s": 7.937578468664457,
        "attn_output_cosine": -0.014532278489024565,
        "head_correlation": 0.0596257838312987
      },
      {
        "student_layer": 5,
        "teacher_layer": 7,
        "cka": 0.941816116198503,
        "principal_angle": 2.1073424255447017e-08,
        "effective_rank_t": 7.911707382811459,
        "effective_rank_s": 7.921112275935485,
        "attn_output_cosine": 0.1267707625569578,
        "head_correlation": 0.0883705381769787
      },
      {
        "student_layer": 6,
        "teacher_layer": 8,
        "cka": 0.957761264502758,
        "principal_angle": 2.1073424255447017e-08,
        "effective_rank_t": 7.919585768861643,
        "effective_rank_s": 7.960201943844822,
        "attn_output_cosine": -0.12865515387069676,
        "head_correlation": 0.06395609310811272
      },
      {
        "student_layer": 7,
        "teacher_layer": 9,
        "cka": 0.9311962371171407,
        "principal_angle": 3.332000937312528e-08,
        "effective_rank_t": 7.94088481581943,
        "effective_rank_s": 7.91150876119655,
        "attn_output_cosine": -0.18589166638589583,
        "head_correlation": 0.06754831994205247
      },
      {
        "student_layer": 8,
        "teacher_layer": 10,
        "cka": 0.9511557683292098,
        "principal_angle": 3.332000937312528e-08,
        "effective_rank_t": 7.930940723029967,
        "effective_rank_s": 7.919732034203233,
        "attn_output_cosine": -0.1288313301341332,
        "head_correlation": 0.06688095521364623
      },
      {
        "student_layer": 9,
        "teacher_layer": 12,
        "cka": 0.9093510835951137,
        "principal_angle": 2.5809568279517847e-08,
        "effective_rank_t": 7.914943158514806,
        "effective_rank_s": 7.932896920855368,
        "attn_output_cosine": -0.06659430517322115,
        "head_correlation": 0.07465597729398975
      },
      {
        "student_layer": 10,
        "teacher_layer": 13,
        "cka": 0.9409860215125533,
        "principal_angle": 2.5809568279517847e-08,
        "effective_rank_t": 7.947400704736904,
        "effective_rank_s": 7.918058142786545,
        "attn_output_cosine": 0.08205179794724701,
        "head_correlation": 0.07268093715120835
      },
      {
        "student_layer": 11,
        "teacher_layer": 14,
        "cka": 0.9376391846396406,
        "principal_angle": 2.1073424255447017e-08,
        "effective_rank_t": 7.91265138399651,
        "effective_rank_s": 7.956349364083351,
        "attn_output_cosine": 0.0922436022128861,
        "head_correlation": 0.08357058799461974
      },
      {
        "student_layer": 12,
        "teacher_layer": 16,
        "cka": 0.9483667445281498,
        "principal_angle": 2.1073424255447017e-08,
        "effective_rank_t": 7.9285690941396325,
        "effective_rank_s": 7.94318942963018,
        "attn_output_cosine": -0.018799930761666885,
        "head_correlation": 0.07168596445574978
      },
      {
        "student_layer": 13,
        "teacher_layer": 17,
        "cka": 0.9121888091965964,
        "principal_angle": 2.5809568279517847e-08,
        "effective_rank_t": 7.9506906237989305,
        "effective_rank_s": 7.89273813758789,
        "attn_output_cosine": 0.029228351656678985,
        "head_correlation": 0.06238034109498438
      },
      {
        "student_layer": 14,
        "teacher_layer": 18,
        "cka": 0.9302857348443528,
        "principal_angle": 3.332000937312528e-08,
        "effective_rank_t": 7.90490473643285,
        "effective_rank_s": 7.943374388460566,
        "attn_output_cosine": -0.10115802855269869,
        "head_correlation": 0.08667054389216759
      },
      {
        "student_layer": 15,
        "teacher_layer": 19,
        "cka": 0.9684952259321129,
        "principal_angle": 1.4901161193847656e-08,
        "effective_rank_t": 7.925036509549946,
        "effective_rank_s": 7.948367653860606,
        "attn_output_cosine": 0.005253166814754963,
        "head_correlation": 0.06270142241042086
      },
      {
        "student_layer": 16,
        "teacher_layer": 21,
        "cka": 0.9184692107806921,
        "principal_angle": 2.5809568279517847e-08,
        "effective_rank_t": 7.917195651937565,
        "effective_rank_s": 7.906697177187549,
        "attn_output_cosine": -0.04134612325692299,
        "head_correlation": 0.06261033634794036
      },
      {
        "student_layer": 17,
        "teacher_layer": 22,
        "cka": 0.9368605728130508,
        "principal_angle": 1.4901161193847656e-08,
        "effective_rank_t": 7.948806854405458,
        "effective_rank_s": 7.922321537065208,
        "attn_output_cosine": 0.037336266192524066,
        "head_correlation": 0.05353006263555638
      },
      {
        "student_layer": 18,
        "teacher_layer": 23,
        "cka": 0.937416162153625,
        "principal_angle": 2.1073424255447017e-08,
        "effective_rank_t": 7.929031805261751,
        "effective_rank_s": 7.8942149034170255,
        "attn_output_cosine": 0.09058234125725913,
        "head_correlation": 0.07504629048709999
      },
      {
        "student_layer": 19,
        "teacher_layer": 25,
        "cka": 0.9444703922269557,
        "principal_angle": 2.5809568279517847e-08,
        "effective_rank_t": 7.929998993716029,
        "effective_rank_s": 7.943379015540937,
        "attn_output_cosine": -0.07009148213952285,
        "head_correlation": 0.07129812370199894
      },
      {
        "student_layer": 20,
        "teacher_layer": 26,
        "cka": 0.9289421426911113,
        "principal_angle": 3.332000937312528e-08,
        "effective_rank_t": 7.931618549997523,
        "effective_rank_s": 7.936433357128866,
        "attn_output_cosine": -0.17912439089393506,
        "head_correlation": 0.08051836292322559
      },
      {
        "student_layer": 21,
        "teacher_layer": 27,
        "cka": 0.9602308642949589,
        "principal_angle": 2.5809568279517847e-08,
        "effective_rank_t": 7.918353525302671,
        "effective_rank_s": 7.935045213856827,
        "attn_output_cosine": 0.055974886361761976,
        "head_correlation": 0.07571432183830944
      },
      {
        "student_layer": 22,
        "teacher_layer": 28,
        "cka": 0.9460370420593497,
        "principal_angle": 2.9802322387695312e-08,
        "effective_rank_t": 7.8918438232265595,
        "effective_rank_s": 7.922253072803384,
        "attn_output_cosine": 0.28651812510803853,
        "head_correlation": 0.0838553635318342
      },
      {
        "student_layer": 23,
        "teacher_layer": 30,
        "cka": 0.9685490331185796,
        "principal_angle": 2.9802322387695312e-08,
        "effective_rank_t": 7.922594945919986,
        "effective_rank_s": 7.947313558092274,
        "attn_output_cosine": 0.11454528613833179,
        "head_correlation": 0.06282543155198614
      },
      {
        "student_layer": 24,
        "teacher_layer": 31,
        "cka": 0.9538096000778685,
        "principal_angle": 3.332000937312528e-08,
        "effective_rank_t": 7.947539254335495,
        "effective_rank_s": 7.942205838257838,
        "attn_output_cosine": -0.05932823559421116,
        "head_correlation": 0.06259542637383667
      },
      {
        "student_layer": 25,
        "teacher_layer": 32,
        "cka": 0.9459982326042137,
        "principal_angle": 2.9802322387695312e-08,
        "effective_rank_t": 7.939897404899496,
        "effective_rank_s": 7.887645992350024,
        "attn_output_cosine": -0.04871717239167139,
        "head_correlation": 0.06154573862655548
      },
      {
        "student_layer": 26,
        "teacher_layer": 34,
        "cka": 0.9283022160782883,
        "principal_angle": 1.4901161193847656e-08,
        "effective_rank_t": 7.924955316437936,
        "effective_rank_s": 7.909932743022746,
        "attn_output_cosine": 0.21297733365574606,
        "head_correlation": 0.07260429991177197
      },
      {
        "student_layer": 27,
        "teacher_layer": 35,
        "cka": 0.9644569200318408,
        "principal_angle": 1.4901161193847656e-08,
        "effective_rank_t": 7.954336399976554,
        "effective_rank_s": 7.913932377218427,
        "attn_output_cosine": -0.13740254838867946,
        "head_correlation": 0.05866295215176825
      }
    ],
    "mean_cka": 0.9433895281768621,
    "mean_principal_angle": 2.548617347935253e-08,
    "mean_attn_output_cosine": -0.009333400034569419,
    "mean_head_correlation": 0.06994921342816911,
    "fig7_axes": {
      "Fig.7a": "layer × layer CKA",
      "Fig.7b": "principal_angle × CHG (T09)",
      "Fig.7c": "attn_output_cosine × retention (T05)",
      "Fig.7d": "effective_rank × CHG"
    }
  }
}
```

## STATISTICAL_CHECK
{
  "n_samples": null,
  "seeds": [
    0,
    1,
    2
  ],
  "ci": 0.95
}

## BEHAVIOR_CHECK
{
  "jcr": "n/a"
}

## GEOMETRY_CHECK
{
  "mean_cka": 0.9433895281768621
}

## SYSTEM_COST
{}

## ACCEPTANCE_CRITERIA
{
  "gate": "OK"
}

## FAILURE_ANALYSIS
(无)

## NEXT_ALLOWED_TASK
t11