/****************************************************************
 * UnifiedSensorLogger.ino
 * 
 * Teensy 4.1 - All sensors logged to SD at 100 Hz
 * 
 * Sensors:
 *   - ADS131M02 (via ADC15 Click) : 2-ch ECG over SPI, running ~500 Hz,
 *     drained continuously, latest values logged at 100 Hz
 *   - ICM-20948 "Chest" : accel + gyro over I2C (Wire,  addr 0x68)
 *   - ICM-20948 "Back"  : accel + gyro over I2C (Wire2, addr 0x69)
 * 
 * SD Logging:
 *   - SdFat with FIFO_SDIO for maximum throughput
 *   - RingBuf decouples sensor reads from SD writes
 *   - File pre-allocated for 10+ hours of recording
 *   - Writes in 512-byte sector-aligned chunks
 * 
 * CSV Format (one row per 100 Hz tick):
 *   millis_since_start,ecg_ch0_raw,ecg_ch1_raw,
 *   chest_ax,chest_ay,chest_az,chest_gx,chest_gy,chest_gz,
 *   back_ax,back_ay,back_az,back_gx,back_gy,back_gz
 * 
 * Architecture:
 *   - IntervalTimer fires at 100 Hz, sets a flag
 *   - Main loop continuously:
 *       1. Drains ECG DRDY (keeps latest sample fresh)
 *       2. On flag: polls both IMUs, formats CSV row into RingBuf
 *       3. Writes RingBuf -> SD in 512-byte chunks when SD not busy
 ****************************************************************/

#include "SdFat.h"
#include "RingBuf.h"
#include "ICM_20948.h"
#include "ADC15_ECG.h"
#include <SPI.h>

// ========================== Configuration ==========================

// Logging rate (Hz) - all sensors logged at this rate
#define LOG_RATE_HZ          100
#define LOG_INTERVAL_US      (1000000 / LOG_RATE_HZ)  // 10000 us

// ECG ADC sample rate (Hz) - runs faster than log rate for freshness
#define ECG_ADC_RATE_HZ      500
#define ECG_ADC_GAIN         GAIN_4   // ±300mV range

// IMU configuration
#define IMU_ACCEL_FSR        gpm8     // ±8g  (same as your DMP config)
#define IMU_GYRO_FSR         dps2000  // ±2000 dps

// I2C buses
#define CHEST_WIRE           Wire     // Chest IMU on Wire  (AD0=0 -> 0x68)
#define BACK_WIRE            Wire2    // Back  IMU on Wire2 (AD0=1 -> 0x69)
#define I2C_CLOCK_HZ         100000   // 100 kHz for long wire runs

// SD card - Teensy 4.1 built-in SDIO
#define SD_CONFIG            SdioConfig(FIFO_SDIO)

// Pre-allocate file for 10 hours at 100 Hz
// ~100 bytes per row * 100 Hz * 3600 s * 10 h = 360 MB
// Round up generously to 512 MB
#define LOG_FILE_SIZE        (512ULL * 1024 * 1024)

// RingBuf: 800 sectors = 400 KB - generous buffer for SD write latency
#define RING_BUF_CAPACITY    (800 * 512)

// Status LED
#define LED_PIN              13

// ========================== Objects ==========================

SdFs sd;
FsFile file;
DMAMEM RingBuf<FsFile, RING_BUF_CAPACITY> rb;

ICM_20948_I2C chestIMU;
ICM_20948_I2C backIMU;
ADC15_ECG ecgADC;

IntervalTimer logTimer;

// ========================== Data Structures ==========================

// Latest ECG readings (updated by continuous DRDY drain)
volatile int32_t latest_ecg_ch0 = 0;
volatile int32_t latest_ecg_ch1 = 0;

// Latest IMU readings (updated at log time)
struct IMUData {
  int16_t ax, ay, az;
  int16_t gx, gy, gz;
};

IMUData chest_data = {0};
IMUData back_data  = {0};

// Timer flag - set by ISR, cleared by main loop
volatile bool logFlag = false;

// Timing
uint32_t session_start_millis = 0;

// Statistics
struct {
  uint32_t rows_written     = 0;
  uint32_t ecg_reads        = 0;
  uint32_t missed_deadlines = 0;
  size_t   maxBytesUsed     = 0;
  uint32_t last_stats_ms    = 0;

  // Loop timing (full loop iteration)
  uint32_t loop_count       = 0;
  uint32_t loop_min_us      = 999999;
  uint32_t loop_max_us      = 0;
  uint32_t loop_sum_us      = 0;

  // Log-tick timing (just the sensor read + RingBuf write portion)
  uint32_t tick_count        = 0;
  uint32_t tick_min_us       = 999999;
  uint32_t tick_max_us       = 0;
  uint32_t tick_sum_us       = 0;

  // SD write timing
  uint32_t sd_write_count    = 0;
  uint32_t sd_write_min_us   = 999999;
  uint32_t sd_write_max_us   = 0;
  uint32_t sd_write_sum_us   = 0;
} stats;

// ========================== ISR ==========================

void logTimerISR() {
  logFlag = true;
}

// ========================== Setup ==========================

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000);

  Serial.println(F("=========================================="));
  Serial.println(F("Unified Sensor Logger - 100 Hz"));
  Serial.println(F("ECG (ADS131M02) + 2x ICM-20948 -> SD"));
  Serial.println(F("=========================================="));

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  // ---------- Initialize SD card ----------
  Serial.print(F("Initializing SD (SDIO)... "));
  if (!sd.begin(SD_CONFIG)) {
    Serial.println(F("FAILED!"));
    sd.initErrorPrint(&Serial);
    fatalError();
  }
  Serial.println(F("OK"));

  // Create unique filename
  char filename[32];
  int file_num = 0;
  do {
    snprintf(filename, sizeof(filename), "log_%03d.csv", file_num++);
  } while (sd.exists(filename));

  if (!file.open(filename, O_RDWR | O_CREAT | O_TRUNC)) {
    Serial.println(F("Failed to create file!"));
    fatalError();
  }
  Serial.print(F("Logging to: "));
  Serial.println(filename);

  // Pre-allocate for 10+ hours
  Serial.print(F("Pre-allocating file... "));
  if (!file.preAllocate(LOG_FILE_SIZE)) {
    Serial.println(F("FAILED!"));
    file.close();
    fatalError();
  }
  Serial.println(F("OK"));

  // Initialize RingBuf
  rb.begin(&file);

  // Write CSV header into the RingBuf
  rb.print(F("millis,ecg_ch0,ecg_ch1,"));
  rb.print(F("chest_ax,chest_ay,chest_az,chest_gx,chest_gy,chest_gz,"));
  rb.println(F("back_ax,back_ay,back_az,back_gx,back_gy,back_gz"));

  // ---------- Initialize I2C ----------
  Serial.println(F("\nInitializing I2C buses..."));
  CHEST_WIRE.begin();
  CHEST_WIRE.setClock(I2C_CLOCK_HZ);
  BACK_WIRE.begin();
  BACK_WIRE.setClock(I2C_CLOCK_HZ);

  // ---------- Initialize Chest IMU ----------
  Serial.print(F("Initializing Chest IMU... "));
  if (!initIMU(chestIMU, CHEST_WIRE, 1)) {  // AD0_VAL=1 for 0x68 on SparkFun lib
    Serial.println(F("FAILED!"));
    fatalError();
  }
  configureIMU(chestIMU);
  Serial.println(F("OK"));

  // ---------- Initialize Back IMU ----------
  Serial.print(F("Initializing Back IMU... "));
  if (!initIMU(backIMU, BACK_WIRE, 1)) {  // AD0_VAL=1 for 0x69
    Serial.println(F("FAILED!"));
    fatalError();
  }
  configureIMU(backIMU);
  Serial.println(F("OK"));

  // ---------- Initialize ECG ADC ----------
  Serial.print(F("Initializing ADS131M02... "));
  if (!ecgADC.begin()) {
    Serial.println(F("FAILED!"));
    fatalError();
  }
  Serial.println(F("OK"));

  Serial.print(F("Configuring ECG for "));
  Serial.print(ECG_ADC_RATE_HZ);
  Serial.print(F(" Hz, gain="));
  Serial.print(ECG_ADC_GAIN);
  Serial.print(F("... "));
  if (!ecgADC.configureSampling(ECG_ADC_RATE_HZ, ECG_ADC_GAIN)) {
    Serial.println(F("FAILED!"));
    fatalError();
  }
  Serial.println(F("OK"));

  uint16_t devID = ecgADC.readID();
  Serial.print(F("ADC Device ID: 0x"));
  Serial.println(devID, HEX);

  // ---------- Start logging ----------
  session_start_millis = millis();
  stats.last_stats_ms = session_start_millis;

  // Start the 100 Hz timer
  logTimer.begin(logTimerISR, LOG_INTERVAL_US);

  Serial.println(F("\n=========================================="));
  Serial.println(F("Logging started at 100 Hz"));
  Serial.println(F("Press 's' for stats, 'q' for quick status"));
  Serial.println(F("==========================================\n"));

  digitalWrite(LED_PIN, HIGH);  // Solid = logging
}

// ========================== Main Loop ==========================

void loop() {
  uint32_t loop_start_us = micros();

  // ---- 1. Continuously drain ECG DRDY for freshest data ----
  //drainECG();

  // ---- 2. On 100 Hz tick: read IMUs, write row to RingBuf ----
  if (logFlag) {
    logFlag = false;

    uint32_t tick_start_us = micros();

    // Read both IMUs (direct register read, no DMP)
    readIMU(chestIMU, chest_data);
    readIMU(backIMU, back_data);

    // Read ECG — if FIFO overflowed, re-sync
    int32_t ch0, ch1;
    if (ecgADC.isDataReady()) {
      if (!ecgADC.readChannels(ch0, ch1)) {
        ecgADC.readChannels(ch0, ch1);  // re-sync
        stats.missed_deadlines++;
      }
      latest_ecg_ch0 = ch0;
      latest_ecg_ch1 = ch1;
      stats.ecg_reads++;
    }

    // Format CSV row into RingBuf
    uint32_t t = millis() - session_start_millis;

    // Use a stack buffer + single rb.write() for efficiency
    char row[160];
    int len = snprintf(row, sizeof(row),
      "%lu,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
      t,
      (long)latest_ecg_ch0, (long)latest_ecg_ch1,
      chest_data.ax, chest_data.ay, chest_data.az,
      chest_data.gx, chest_data.gy, chest_data.gz,
      back_data.ax,  back_data.ay,  back_data.az,
      back_data.gx,  back_data.gy,  back_data.gz
    );

    if (rb.write(row, len) != (size_t)len) {
      // RingBuf overrun - should not happen with proper sizing
      stats.missed_deadlines++;
    } else {
      stats.rows_written++;
    }

    // Track max RingBuf usage
    size_t used = rb.bytesUsed();
    if (used > stats.maxBytesUsed) {
      stats.maxBytesUsed = used;
    }

    // Tick timing
    uint32_t tick_us = micros() - tick_start_us;
    stats.tick_count++;
    stats.tick_sum_us += tick_us;
    if (tick_us < stats.tick_min_us) stats.tick_min_us = tick_us;
    if (tick_us > stats.tick_max_us) stats.tick_max_us = tick_us;
  }

  // ---- 3. Drain RingBuf to SD in 512-byte sectors ----
  size_t n = rb.bytesUsed();
  if (n >= 512 && !file.isBusy()) {
    uint32_t sd_start_us = micros();
    if (512 != rb.writeOut(512)) {
      Serial.println(F("ERROR: writeOut failed!"));
    }
    uint32_t sd_us = micros() - sd_start_us;
    stats.sd_write_count++;
    stats.sd_write_sum_us += sd_us;
    if (sd_us < stats.sd_write_min_us) stats.sd_write_min_us = sd_us;
    if (sd_us > stats.sd_write_max_us) stats.sd_write_max_us = sd_us;
  }
  // Periodic sync so power loss only loses ~60s of data
  static uint32_t last_sync = 0;
  if ((millis() - last_sync > 60000) && !file.isBusy()) {
    rb.sync();
    file.flush();
    last_sync = millis();
  }

  // ---- 4. Check for file full ----
  if ((rb.bytesUsed() + file.curPosition()) > (LOG_FILE_SIZE - 512)) {
    stopLogging("File full");
  }

  // ---- 5. Serial commands (non-blocking) ----
  if (Serial.available()) {
    char cmd = Serial.read();
    while (Serial.available()) Serial.read();  // flush

    switch (cmd) {
      case 's': case 'S': printStats();       break;
      case 'q': case 'Q': printQuickStatus(); break;
      case 'x': case 'X': stopLogging("User requested stop"); break;
    }
  }

  // ---- Loop timing ----
  uint32_t loop_us = micros() - loop_start_us;
  stats.loop_count++;
  stats.loop_sum_us += loop_us;
  if (loop_us < stats.loop_min_us) stats.loop_min_us = loop_us;
  if (loop_us > stats.loop_max_us) stats.loop_max_us = loop_us;
}

// ========================== IMU Functions ==========================

bool initIMU(ICM_20948_I2C &imu, TwoWire &wire, uint8_t ad0val) {
  int retries = 0;
  while (imu.begin(wire, ad0val) != ICM_20948_Stat_Ok) {
    delay(500);
    if (++retries > 10) return false;
  }
  return true;
}

void configureIMU(ICM_20948_I2C &imu) {
  // Software reset for clean state
  imu.swReset();
  delay(250);

  // Wake up
  imu.sleep(false);
  imu.lowPower(false);

  // Continuous sampling mode for accel and gyro
  imu.setSampleMode(
    (ICM_20948_Internal_Acc | ICM_20948_Internal_Gyr),
    ICM_20948_Sample_Mode_Continuous
  );

  // Full-scale ranges (matching your previous DMP config)
  ICM_20948_fss_t fss;
  fss.a = IMU_ACCEL_FSR;  // gpm8  = ±8g
  fss.g = IMU_GYRO_FSR;   // dps2000 = ±2000 dps
  imu.setFullScale((ICM_20948_Internal_Acc | ICM_20948_Internal_Gyr), fss);

  // Set sample rate divider for ~100 Hz
  // ICM-20948 base rate = 1125 Hz for accel/gyro
  // Divider = (1125 / desired_rate) - 1 = (1125/100) - 1 ≈ 10
  // Actual rate = 1125 / (10+1) = 102.27 Hz
  ICM_20948_smplrt_t smplrt;
  smplrt.a = 1;  // accel: 1125/(1+1) ≈ 562.5 Hz
  smplrt.g = 1;  // gyro:  1125/(1+1) ≈ 562.5 Hz
  imu.setSampleRate(
    (ICM_20948_Internal_Acc | ICM_20948_Internal_Gyr),
    smplrt
  );

  // Digital low-pass filter: ~50 Hz bandwidth
  // Good anti-alias for 100 Hz logging
  ICM_20948_dlpcfg_t dlp;
  dlp.a = acc_d50bw4_n68bw8;     // Accel: 50.4 Hz BW
  dlp.g = gyr_d51bw2_n73bw3;     // Gyro:  51.2 Hz BW
  imu.setDLPFcfg((ICM_20948_Internal_Acc | ICM_20948_Internal_Gyr), dlp);

  // Enable DLPF
  imu.enableDLPF(ICM_20948_Internal_Acc, true);
  imu.enableDLPF(ICM_20948_Internal_Gyr, true);
}

void readIMU(ICM_20948_I2C &imu, IMUData &data) {
  if (imu.dataReady()) {
    imu.getAGMT();  // Reads all sensor registers at once
    data.ax = imu.agmt.acc.axes.x;
    data.ay = imu.agmt.acc.axes.y;
    data.az = imu.agmt.acc.axes.z;
    data.gx = imu.agmt.gyr.axes.x;
    data.gy = imu.agmt.gyr.axes.y;
    data.gz = imu.agmt.gyr.axes.z;
  }
}

// ========================== Logging Control ==========================

void stopLogging(const char* reason) {
  logTimer.end();  // Stop the 100 Hz timer

  // Flush all remaining RingBuf data to file
  rb.sync();
  file.truncate();  // Trim pre-allocated space to actual size
  file.close();

  Serial.println();
  Serial.print(F("Logging stopped: "));
  Serial.println(reason);
  printStats();

  // Blink slowly to indicate stopped
  while (true) {
    digitalWrite(LED_PIN, HIGH);
    delay(500);
    digitalWrite(LED_PIN, LOW);
    delay(500);

    // Allow restart with 'r'
    if (Serial.available() && Serial.read() == 'r') {
      Serial.println(F("Resetting..."));
      SCB_AIRCR = 0x05FA0004;  // Teensy software reset
    }
  }
}

// ========================== Status / Debug ==========================

void printStats() {
  uint32_t elapsed_ms = millis() - session_start_millis;
  float elapsed_sec = elapsed_ms / 1000.0f;

  Serial.println(F("\n========== Statistics =========="));
  Serial.print(F("Elapsed: "));
  Serial.print(elapsed_sec, 1);
  Serial.println(F(" s"));

  Serial.print(F("Rows written: "));
  Serial.println(stats.rows_written);

  if (elapsed_sec > 0) {
    Serial.print(F("Effective log rate: "));
    Serial.print(stats.rows_written / elapsed_sec, 1);
    Serial.println(F(" Hz"));

    Serial.print(F("ECG drain rate: "));
    Serial.print(stats.ecg_reads / elapsed_sec, 1);
    Serial.println(F(" Hz"));
  }

  Serial.print(F("Missed deadlines: "));
  Serial.println(stats.missed_deadlines);

  // RingBuf usage
  Serial.print(F("\nRingBuf max used: "));
  Serial.print(stats.maxBytesUsed);
  Serial.print(F(" / "));
  Serial.print(RING_BUF_CAPACITY);
  Serial.print(F(" ("));
  Serial.print(100.0f * stats.maxBytesUsed / RING_BUF_CAPACITY, 1);
  Serial.println(F("%)"));

  Serial.print(F("File position: "));
  Serial.print((uint32_t)(file.curPosition() / 1024));
  Serial.println(F(" KB"));

  // Loop timing
  if (stats.loop_count > 0) {
    uint32_t loop_avg = stats.loop_sum_us / stats.loop_count;
    Serial.println(F("\n-- Full Loop Timing --"));
    Serial.print(F("  Iterations: "));
    Serial.println(stats.loop_count);
    Serial.print(F("  Min: "));
    Serial.print(stats.loop_min_us);
    Serial.println(F(" us"));
    Serial.print(F("  Avg: "));
    Serial.print(loop_avg);
    Serial.println(F(" us"));
    Serial.print(F("  Max: "));
    Serial.print(stats.loop_max_us);
    Serial.println(F(" us"));
    Serial.print(F("  Loops/tick: ~"));
    Serial.println(stats.loop_count / max(stats.tick_count, (uint32_t)1));
  }

  // Tick timing (sensor read + ringbuf write on 100 Hz tick)
  if (stats.tick_count > 0) {
    uint32_t tick_avg = stats.tick_sum_us / stats.tick_count;
    Serial.println(F("\n-- 100 Hz Tick Timing (IMU read + format + RingBuf write) --"));
    Serial.print(F("  Ticks: "));
    Serial.println(stats.tick_count);
    Serial.print(F("  Min: "));
    Serial.print(stats.tick_min_us);
    Serial.println(F(" us"));
    Serial.print(F("  Avg: "));
    Serial.print(tick_avg);
    Serial.println(F(" us"));
    Serial.print(F("  Max: "));
    Serial.print(stats.tick_max_us);
    Serial.println(F(" us"));

    // Margin: how much of the 10ms budget are we using?
    float margin_pct = 100.0f * (1.0f - (float)tick_avg / LOG_INTERVAL_US);
    Serial.print(F("  Budget margin: "));
    Serial.print(margin_pct, 1);
    Serial.println(F("% (of 10000 us)"));

    if (stats.tick_max_us > LOG_INTERVAL_US) {
      Serial.print(F("  WARNING: Max tick ("));
      Serial.print(stats.tick_max_us);
      Serial.print(F(" us) exceeds budget by "));
      Serial.print(stats.tick_max_us - LOG_INTERVAL_US);
      Serial.println(F(" us!"));
    }
  }

  // SD write timing
  if (stats.sd_write_count > 0) {
    uint32_t sd_avg = stats.sd_write_sum_us / stats.sd_write_count;
    Serial.println(F("\n-- SD Write Timing (per 512-byte sector) --"));
    Serial.print(F("  Writes: "));
    Serial.println(stats.sd_write_count);
    Serial.print(F("  Min: "));
    Serial.print(stats.sd_write_min_us);
    Serial.println(F(" us"));
    Serial.print(F("  Avg: "));
    Serial.print(sd_avg);
    Serial.println(F(" us"));
    Serial.print(F("  Max: "));
    Serial.print(stats.sd_write_max_us);
    Serial.println(F(" us"));
  }

  Serial.println(F("================================\n"));
}

void printQuickStatus() {
  uint32_t elapsed_sec = (millis() - session_start_millis) / 1000;
  uint32_t tick_avg = (stats.tick_count > 0) ? stats.tick_sum_us / stats.tick_count : 0;
  Serial.print(elapsed_sec);
  Serial.print(F("s | "));
  Serial.print(stats.rows_written);
  Serial.print(F(" rows | tick avg "));
  Serial.print(tick_avg);
  Serial.print(F("us max "));
  Serial.print(stats.tick_max_us);
  Serial.print(F("us | buf "));
  Serial.print(100.0f * rb.bytesUsed() / RING_BUF_CAPACITY, 0);
  Serial.print(F("% | miss "));
  Serial.println(stats.missed_deadlines);
}

// ========================== Error Handler ==========================

void fatalError() {
  Serial.println(F("\n*** FATAL ERROR - System halted ***"));
  while (true) {
    digitalWrite(LED_PIN, HIGH);
    delay(100);
    digitalWrite(LED_PIN, LOW);
    delay(100);
  }
}
