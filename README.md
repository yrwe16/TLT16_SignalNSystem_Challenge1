# 저희가왜마지막이조(TLT) · Speech Noise Reduction Challenge

| 항목 | 내용 |
| --- | --- |
| 팀명 | 저희가왜마지막이조(TLT, TheLastTeam(16조)) |
| 제출 회차 | 1회차 |
| 평가 버전 | Git 태그 `round-01` (커밋 해시는 런어스 제출 글에 함께 기재) |
| 알고리즘 | Wiener Filter (decision-directed 사전 SNR 추정), 딥러닝·머신러닝 미사용 |

## 실행 방법

Python 3.10 이상에서 동작합니다 (Python 3.10, 3.12, 3.13, 3.14에서 확인).

```bash
# 1. 패키지 설치
pip install -r requirements.txt

# 2. 테스트 WAV 파일을 input/ 폴더에 넣고 실행
python denoise.py

# 3. 결과는 output/ 폴더에 같은 파일명으로 저장됩니다
```

다른 폴더를 쓰려면 `python denoise.py --input <입력 폴더> --output <출력 폴더>`로 지정합니다.
경로는 모두 저장소 기준 상대 경로이며, 개인 PC의 절대 경로에 의존하지 않습니다.

### 입출력 규칙

- `input/` 아래의 모든 `.wav` 파일(대소문자 무관, 하위 폴더 포함)을 처리합니다.
- 입력 파일마다 **같은 파일명**의 출력 WAV 하나를 `output/`에 만듭니다.
- 출력은 입력과 **같은 샘플링 주파수, 채널 수, 샘플 수, WAV 형식**(예: 16-bit PCM)을 유지합니다.
- 여러 채널이면 채널별로 따로 처리합니다. 16 kHz가 아닌 입력은 창 길이(32 ms)가 같도록 STFT 크기를 맞춥니다.
- 정수 PCM 출력의 피크가 1을 넘으면 클리핑 대신 파일 전체 레벨을 일정하게 낮춥니다.
- 읽을 수 없는 파일이 있어도 중단하지 않고, 경고를 표시한 뒤 입력을 그대로 복사해 출력 파일이 빠지지 않게 합니다.

결과가 규칙을 지키는지 확인하려면 다음을 실행합니다.

```bash
python tools/check_io.py          # input/ 과 output/ 비교
```

## 알고리즘

강의에서 제공한 베이스라인 노트북(`noise_reduction_baselines.ipynb`)의 Wiener Filter를 제출 형식에 맞게 옮긴 것입니다.
16 kHz 입력에서 노트북의 `wiener_filter` 함수와 출력이 샘플 단위로 동일함을 확인했습니다.

1. **STFT:** Hann 창 512 샘플(32 ms), hop 128 샘플(75 % 겹침).
2. **잡음 파워 추정:** 경계 패딩을 제외한 완전한 프레임 중 에너지가 가장 낮은 20 %의 평균 파워를 주파수 방향으로 3-bin 평활화합니다. 파일마다 하나의 값을 사용하며 Noisy 입력만 사용합니다.
3. **이득 계산:** 사후 SNR $\gamma_t = |Y_t|^2 / N$, 사전 SNR $\xi_t = a\,|\hat S_{t-1}|^2/N + (1-a)\max(\gamma_t-1, 0)$ (decision-directed).
   이득 규칙은 `gain_rule`로 고르며, 이득을 곱한 뒤 입력 위상으로 복원합니다.
   - `wiener` (현재 기본값, 1회차 제출본): $G_t = \mathrm{clip}\big(\xi_t/(1+\xi_t),\ g_{\min},\ 1\big)$
   - `lsa` (다음 회차 후보): MMSE-LSA $G_t = \max\big(\tfrac{\xi_t}{1+\xi_t}\exp(\tfrac12 E_1(v_t)),\ g_{\min}\big)$, $v_t = \tfrac{\xi_t}{1+\xi_t}\gamma_t$ (Ephraim & Malah, 1985). 논문처럼 상한은 두지 않습니다.
4. **ISTFT:** overlap-add로 파형을 복원하고 입력 길이에 맞춥니다.

| 파라미터 | 값 | 위치 |
| --- | --- | --- |
| 창 길이 / 겹침 | 32 ms / 75 % | `denoiser/config.py` |
| 잡음 추정 프레임 비율 | 0.20 | `noise_fraction` |
| 이득 규칙 | `wiener` (다음 회차 후보는 `lsa`) | `gain_rule` |
| LSA decision-directed 계수 $a$ | 0.95 | `lsa_smoothing` |
| LSA 최소 이득 $g_{\min}$ | 0.178 (−15 dB) | `lsa_gain_floor` |
| Wiener decision-directed 계수 $a$ | 0.98 | `wiener_smoothing` |
| Wiener 최소 이득 $g_{\min}$ | 0.05 (−26 dB) | `wiener_gain_floor` |
| 잡음 추정 방식 | `fixed` (단계 2 실험은 `spp`, 미채택) | `noise_tracker`, `spp_*` |

LSA 파라미터는 Train에서 PESQ-WB 기준으로 골랐습니다 (최소 이득 −10/−12/−15/−20/−25 dB × $a$ 0.92/0.95/0.98).
`gain_rule="lsa"`로 바꾸면 1단계 실험(MMSE-LSA)의 출력을 냅니다. 이 스위치를 추가해도 `wiener` 출력은 이전과 바이트 단위로 같습니다.

모든 파라미터는 코드(`denoiser/config.py`)에 포함되어 있으며 별도 설정 파일이 필요 없습니다.

## Validation 평가 결과

1회차(`round-01`) 제출 버전(Wiener)으로 Validation 164개 파일을 평가한 결과입니다 (`results/validation_summary.csv`).
MMSE-LSA(`gain_rule="lsa"`)는 아직 제출하지 않은 실험 결과이며, 아래 "실험 기록"에 따로 정리했습니다.

| 입력 | 파일 수 | 평균 SNR (dB) | 평균 PESQ-WB |
| --- | ---: | ---: | ---: |
| Noisy (처리 전) | 164 | 9.63 | 1.395 |
| 제출 알고리즘 (Wiener) | 164 | 13.12 | 1.599 |
| 개선량 | | +3.49 | +0.204 |

PESQ 계산 실패 파일은 0개입니다. 잡음 종류·목표 SNR·난이도별 결과는 `results/validation_by_*.csv`, 파일별 점수는 `results/validation_per_file.csv`에 있습니다.

평가 방법은 베이스라인 노트북과 같습니다.

- **평가 구간:** `samples.csv`에 `valid_samples` 열이 있으면 노트북처럼 앞에서부터 그 길이까지만 채점하고(잡음 제거는 파일 전체에 적용), 없으면 파일 전체를 채점합니다. 현재 Validation 데이터에는 이 열이 없어 파일 전체를 채점했습니다.
- **SNR:** $10\log_{10}\big(\sum s^2 / \sum (s-\hat s)^2\big)$, 레벨·시간 정렬 없음.
- **PESQ:** 16 kHz wideband (`pesq` 0.0.4, mode `wb`).
- **집계:** 파일별 점수의 단순 평균.

`denoise.py`가 실제로 저장한 WAV 파일을 채점하므로, 16-bit 저장 과정의 영향도 포함됩니다.

### 평가 재현

```bash
pip install -r requirements-eval.txt
python tools/evaluate.py --data-root <noise_reduction_challenge 폴더>
```

`<noise_reduction_challenge 폴더>` 안에 `validation/samples.csv`가 있어야 합니다.
결과는 `results/`에 저장됩니다 (전체 요약, 파일별 점수, 잡음 종류·SNR·난이도별 요약).
Windows에서 `pesq` 설치가 실패하면 `conda install -c conda-forge pesq=0.0.4`를 사용하세요.

## 실험 기록

단계별 실험은 `results/experiments.csv`에 기록하고, 실험별 Train 후보·Validation 결과·짝지은 검정·스펙트로그램은 `results/experiments/<날짜>-<이름>/`에 둡니다.

| 단계 | 변경 | Train PESQ-WB | Validation PESQ-WB | Validation SNR (dB) | 결정 |
| --- | --- | ---: | ---: | ---: | --- |
| 0 | Wiener (1회차 제출본) | 1.604 | 1.599 | 13.12 | 기준 |
| 1 | + 이득 함수 MMSE-LSA (−15 dB, $a$ = 0.95) | 1.763 | 1.730 | 13.57 | 채택, 다음 회차 제출 예정 (Wilcoxon $p$ = 7×10⁻¹³) |
| 2 | + 잡음 추적 MMSE-SPP (양방향, ξ_H1 = 10 dB) | 1.779 | 1.741 | 13.15 | 미채택 (Validation $p$ = 0.56, aircraft_cockpit −0.11) |

단계 2는 1단계 설정 위에 잡음 PSD를 프레임마다 추적하는 unbiased MMSE-SPP(Gerkmann & Hendriks, 2012)를 더한 실험입니다 (`noise_tracker="spp"`, 기본값은 `"fixed"`).
논문 그대로의 인과 추적(VOICEBOX 기본값)은 Train에서 1단계보다 0.064 낮았고, 비인과(양방향) 변형만 Train에서 +0.016 올랐습니다.
Validation에서는 유의하지 않았고 정상 잡음(조종석·광대역)이 나빠져 채택하지 않았습니다. 바람 잡음은 두 데이터 모두에서 좋아졌습니다 (Validation +0.068).

```bash
# 설정을 바꿔 평가 (결과를 실험 폴더에 저장)
python tools/evaluate.py --data-root <데이터 폴더> --split train --set gain_rule=lsa \
    --set lsa_gain_floor=0.178 --results-dir results/experiments/<이름>/train-<후보>
# 두 평가 결과의 파일별 짝지은 비교 (Wilcoxon signed-rank, 잡음 종류·SNR별)
python tools/compare_runs.py <기준 결과 폴더> <새 결과 폴더> --split validation
# 고정 청취 목록(results/listening_set.txt)의 스펙트로그램 (matplotlib 필요)
python tools/plot_spectrograms.py --data-root <데이터 폴더> --run "이전=<폴더>" --run "새=<폴더>" --out <폴더>
```

## 폴더 구조

```
denoise.py              실행 파일 (input/ → output/)
denoiser/
  config.py             모든 파라미터
  stft.py               STFT / ISTFT
  noise_estimation.py   잡음 파워 추정 (파일 고정값 / MMSE-SPP 추적)
  gains.py              Wiener / MMSE-LSA 이득 (decision-directed)
  pipeline.py           신호·파일·폴더 단위 처리
tools/
  check_io.py           입출력 규칙 검사
  evaluate.py           Train / Validation SNR / PESQ 평가 (--set 으로 설정 덮어쓰기)
  compare_runs.py       두 평가 결과의 짝지은 검정, 잡음 종류·SNR별 비교
  plot_spectrograms.py  고정 청취 목록의 스펙트로그램
requirements.txt        실행에 필요한 패키지
requirements-eval.txt   평가에만 필요한 패키지
input/  output/         테스트 입력 / 결과 폴더
```

## 참고 문헌

- Y. Ephraim and D. Malah, "Speech Enhancement Using a Minimum Mean-Square Error Short-Time Spectral Amplitude Estimator," *IEEE Transactions on Acoustics, Speech, and Signal Processing*, vol. 32, no. 6, pp. 1109–1121, 1984. (decision-directed 사전 SNR 추정)
- Y. Ephraim and D. Malah, "Speech Enhancement Using a Minimum Mean-Square Error Log-Spectral Amplitude Estimator," *IEEE Transactions on Acoustics, Speech, and Signal Processing*, vol. 33, no. 2, pp. 443–445, 1985, DOI 10.1109/TASSP.1985.1164550. (MMSE-LSA 이득)
- T. Gerkmann and R. C. Hendriks, "Unbiased MMSE-Based Noise Power Estimation With Low Complexity and Low Tracking Delay," *IEEE Transactions on Audio, Speech, and Language Processing*, vol. 20, no. 4, pp. 1383–1393, 2012, DOI 10.1109/TASL.2011.2180896. (단계 2 실험, 미채택)
- 이득 수식 대조용으로 읽은 코드 (복사하지 않고 직접 구현): wilsonchingg/logmmse `logmmse/logmmse.py` (MIT), ImperialCollegeLondon/sap-voicebox `voicebox/v_ssubmmse.m`, `voicebox/v_estnoiseg.m` (LGPL-3, 단계 2 상수의 출처).
- 코드 출처: 강의 제공 베이스라인 노트북 `noise_reduction_baselines.ipynb`의 `stft_audio`, `inverse_stft`, `estimate_noise_power`, `wiener_filter`.
