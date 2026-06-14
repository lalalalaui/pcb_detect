#ifndef PCB_TINY_AE_THRESHOLDS_H
#define PCB_TINY_AE_THRESHOLDS_H

/*
 * TinyAE anomaly threshold calibrated from the full validation set:
 *   data/processed/pcb_anomaly/val/normal
 *   data/processed/pcb_anomaly/val/anomaly
 *
 * This threshold is not derived from the 6-sample smoke test vectors.
 * BEST_F1 is a high-recall threshold and may cause many false positives.
 * NORMAL_P99 is the low-false-positive strong alarm threshold.
 * BALANCED can be used for NORMAL/SUSPECT/ANOMALY three-level display.
 * TINY_AE_THRESHOLD defaults to NORMAL_P99 for STM32 strong alarm use.
 * If the model is retrained, quantized, or input preprocessing changes,
 * recalibrate this file before deploying.
 * STM32 firmware must use the same RGB 96x96 float32 [0,1] NCHW
 * preprocessing and mean((input - output)^2) MSE calculation.
 */
#define TINY_AE_THRESHOLD_BEST_F1      8.04748561e-05f
#define TINY_AE_THRESHOLD_BALANCED     0.000297238343f
#define TINY_AE_THRESHOLD_NORMAL_P95   0.000447257333f
#define TINY_AE_THRESHOLD_NORMAL_P99   0.000630778263f

#define TINY_AE_THRESHOLD_LOW          TINY_AE_THRESHOLD_BALANCED
#define TINY_AE_THRESHOLD_HIGH         TINY_AE_THRESHOLD_NORMAL_P99
#define TINY_AE_THRESHOLD              TINY_AE_THRESHOLD_HIGH

#endif /* PCB_TINY_AE_THRESHOLDS_H */
