from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Имитируем структуру путей, чтобы не ломать логику лабы
DATA = Path("data")

# ==========================================
# Задания 1 и 2. Автономная генерация векторов сигналов
# ==========================================
# Размерность 120000 строго соответствует условиям вашей лабораторной работы
np.random.seed(42)  # Фиксируем случайность для воспроизводимости
normal = np.random.normal(0, 1.0, 120000)
roller = np.random.normal(0, 1.5, 120000)
inner = np.random.normal(0, 1.8, 120000)
outer = np.random.normal(0, 2.0, 120000)

print(f"Сигналы успешно сгенерированы. Длина векторов: {len(normal)} отсчётов.")


# ==========================================
# Задание 3. Окна с перекрытием
# ==========================================
def segment_signal(signal, length=4000, overlap=0.75):
    step = int(length * (1 - overlap))  # Шаг окна (4000 * 0.25 = 1000)
    windows = []
    for start in range(0, len(signal) - length + 1, step):
        windows.append(signal[start:start + length])
    return np.stack(windows)


segments_normal = segment_signal(normal)
segments_roller = segment_signal(roller)
segments_inner = segment_signal(inner)
segments_outer = segment_signal(outer)

# ==========================================
# Задание 4. Синтетические смеси сигналов
# ==========================================
mixing_coefficients = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]


def rms(signal):
    centered = signal - np.mean(signal)
    return np.sqrt(np.mean(centered ** 2))


X_values = [rms(normal)]
y_values = [[0.0, 0.0, 0.0]]

for target_col, defect in enumerate([inner, outer, roller]):
    for mixing_coefficient in mixing_coefficients:
        mixture = normal + mixing_coefficient * (defect - normal)
        X_values.append(rms(mixture))

        target = np.zeros(3)
        target[target_col] = mixing_coefficient * 100.0
        y_values.append(target)

X_matrix = np.array(X_values).reshape(-1, 1)
y_matrix = np.array(y_values)

# ==========================================
# Задание 5. Четырнадцать признаков одного окна
# ==========================================
N, fs = 4000, 10_000
x = normal[:N] - np.mean(normal[:N])

rms_value = np.sqrt(np.mean(x ** 2))

amplitude = np.abs(np.fft.rfft(x))[:-1]
frequency = np.fft.rfftfreq(N, d=1 / fs)[:-1]

dominant_frequency = frequency[np.argmax(amplitude)]

band_energy = []
for i in range(10):
    flow = i * 500
    fhigh = (i + 1) * 500
    idx = (frequency >= flow) & (frequency < fhigh)
    band_energy.append(np.sum(amplitude[idx] ** 2))

zero_crossing_rate = np.mean(np.diff(np.sign(x)) != 0)

q25 = np.percentile(np.abs(x), 25)
q75 = np.percentile(np.abs(x), 75)
ratio_q = q75 / q25 if q25 != 0 else q75

features = [rms_value, dominant_frequency] + band_energy + [zero_crossing_rate, ratio_q]
features = np.array(features)

# ==========================================
# Задание 6. Нормализация признаков
# ==========================================
means = np.mean(X_matrix, axis=0)
stds = np.std(X_matrix, axis=0, ddof=1)
stds[stds < 1e-10] = 1.0
X_scaled = (X_matrix - means) / stds

# ==========================================
# Задание 7. Создать три случайных леса
# ==========================================
model_inner = RandomForestRegressor(n_estimators=400, max_depth=35, min_samples_split=2, min_samples_leaf=1,
                                    random_state=42)
model_outer = RandomForestRegressor(n_estimators=400, max_depth=35, min_samples_split=2, min_samples_leaf=1,
                                    random_state=43)
model_roller = RandomForestRegressor(n_estimators=600, max_depth=50, min_samples_split=2, min_samples_leaf=1,
                                     random_state=44)

print("Все расчеты и инициализации моделей успешно завершены!")
