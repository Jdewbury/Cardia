/****************************************************************
 * UnifiedSensorLogger.ino
 * 
 * Teensy 4.1 - All sensors logged to SD at 100 Hz
 * 
 * Sensors:
 *   - ADS131M02 (via ADC15 Click) : 2-ch ECG over SPI, running ~500 Hz,
 *     drained continuously, latest values logged at 100 Hz
 *   - ICM-42670-P "Chest" : accel + gyro over I2C (Wire,  addr 0x69)
 *   - ICM-20948 "Back"  : accel + gyro over I2C (Wire2, addr 0x69)
 * 
 * SD Logging:
 *   - SdFat with FIFO_SDIO for maximum throughput
 *   - RingBuf decouples sensor reads from SD writes
 *   - File pre-allocated for 10+ hours of recording
 *   - Writes in 512-byte sector-aligned chunks
 * 
 * CSV Format (one row per 100 Hz tick):
 *   timestamp,ecg_leadIII,ecg_leadI,
 *   chest_ax,chest_ay,chest_az,chest_gx,chest_gy,chest_gz,
 *   back_ax,back_ay,back_az,back_gx,back_gy,back_gz
 * 
 * Architecture:
 *   - IntervalTimer fires at 100 Hz, sets a flag
 *   - Main loop continuously:
 *       1. On flag: polls both IMUs and the ADC, formats CSV row into RingBuf
 *       2. Writes RingBuf -> SD in 512-byte chunks when SD not busy
 ****************************************************************/

#include "SdFat.h"
#include "RingBuf.h"
#include "ICM_20948.h"
#include "ICM42670P.h"
#include "ADC15_ECG.h"
#include <SPI.h>
#include <TimeLib.h>

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
#define CHEST_ACCEL_FSR      8
#define CHEST_GYRO_FSR       2000

// I2C buses
#define CHEST_WIRE           Wire     // Chest IMU on Wire  (AD0=1 -> 0x69)
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
FsFile logFile;
DMAMEM RingBuf<FsFile, RING_BUF_CAPACITY> rb;

ICM42670 chestIMU(CHEST_WIRE, 1);
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

// ========================== ISR ==========================

void logTimerISR() {
  logFlag = true;
}

// ========================== Setup ==========================

void setup() {
  while (millis() < 3000);
  setSyncProvider(getTeensy3Time);

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  
  // Enable TXS0104 Pin for level shifting
  pinMode(7, OUTPUT);
  delay(100);
  digitalWrite(7, HIGH);

  // ---------- Initialize SD card ----------
  if (!sd.begin(SD_CONFIG)) {
    sd.initErrorPrint(&Serial);
    fatalError();
  }
  if (logFile.open("logs.txt", O_RDWR | O_CREAT | O_TRUNC )) {
    logFile.print(day()); 
    logFile.print(" "); 
    logFile.print(month()); 
    logFile.print(" ");
    logFile.println(year()); 
    logFile.print("Log file of most recent run");
    logFile.flush();
  }
  // Create unique data file filename
  char filename[32];
  snprintf(filename, sizeof(filename), "%04d%02d%02d_%02d%02d%02d.csv",
    year(), month(), day(), hour(), minute(), second());

  // Fallback with counter if filename already exists (e.g. two boots in same second)
  if (sd.exists(filename)) {
    int file_num = 1;
    do {
      snprintf(filename, sizeof(filename), "%04d%02d%02d_%02d%02d%02d_%d.csv",
        year(), month(), day(), hour(), minute(), second(), file_num++);
    } while (sd.exists(filename));
  }

  if (!file.open(filename, O_RDWR | O_CREAT | O_TRUNC)) {
    logFile.println(F("Failed to create file!"));
    logFile.flush();
    fatalError();
  }
  logFile.print(F("Logging to: "));
  logFile.println(filename);
  logFile.flush();

  // Pre-allocate for 10+ hours
  logFile.print(F("Pre-allocating file... "));
  if (!file.preAllocate(LOG_FILE_SIZE)) {
    logFile.println(F("FAILED!"));
    logFile.flush();
    file.close();
    fatalError();
  }
  logFile.println(F("OK"));
  logFile.flush();

  // Initialize RingBuf
  rb.begin(&file);

  // Write CSV header into the RingBuf
  rb.print(F("timestamp,ecg_leadIII,ecg_leadI,"));
  rb.print(F("chest_ax,chest_ay,chest_az,chest_gx,chest_gy,chest_gz,"));
  rb.println(F("back_ax,back_ay,back_az,back_gx,back_gy,back_gz"));

  // ---------- Initialize I2C ----------
  CHEST_WIRE.begin();
  CHEST_WIRE.setClock(I2C_CLOCK_HZ);
  BACK_WIRE.begin();
  BACK_WIRE.setClock(I2C_CLOCK_HZ);

  // ---------- Initialize Chest IMU ----------
  logFile.print(F("Initializing Chest IMU... "));
  logFile.flush();
  bool chestStat;
  chestStat = initChestIMU(chestIMU);
  if (!chestStat) {  
    logFile.println(F("FAILED!"));
    fatalError();
  }
  logFile.println(F("PASSED"));
  logFile.print("Configuring Chest IMU... ");
  logFile.flush();
  configureChestIMU(chestIMU);
  logFile.println("OK");

  // ---------- Initialize Back IMU ----------
  logFile.print(F("Initializing Back IMU... "));
  logFile.flush();
  if (!initIMU(backIMU, BACK_WIRE, 1)) {  // AD0_VAL=1 for 0x69
    logFile.println(F("FAILED!"));
    fatalError();
  }
  logFile.println("PASSED");
  logFile.print("Configuring Back IMU... ");
  logFile.flush();
  configureIMU(backIMU);
  logFile.println(F("OK"));

  // ---------- Initialize ECG ADC ----------
  logFile.print(F("Initializing ADS131M02... "));
  logFile.flush();
  if (!ecgADC.begin()) {
    logFile.println(F("FAILED!"));
    fatalError();
  }
  logFile.println(F("OK"));

  logFile.print(F("Configuring ECG for "));
  logFile.print(ECG_ADC_RATE_HZ);
  logFile.print(F(" Hz, gain="));
  logFile.print(ECG_ADC_GAIN);
  logFile.print(F("... "));
  logFile.flush();
  if (!ecgADC.configureSampling(ECG_ADC_RATE_HZ, ECG_ADC_GAIN)) {
    Serial.println(F("FAILED!"));
    fatalError();
  }
  logFile.println(F("OK"));
  logFile.flush();
  uint16_t devID = ecgADC.readID();
  logFile.print(F("ADC Device ID: 0x"));
  logFile.println(devID, HEX);
  logFile.flush();

  // ---------- Start logging ----------
  session_start_millis = millis();

  // Start the 100 Hz timer
  logTimer.begin(logTimerISR, LOG_INTERVAL_US);

  logFile.println(F("\n=========================================="));
  logFile.println(F("Logging started at 100 Hz"));
  logFile.println(F("Press 's' for stats, 'q' for quick status"));
  logFile.println(F("==========================================\n"));
  logFile.close();
}

// ========================== Main Loop ==========================

void loop() {
  // ---- 1. On 100 Hz tick: read IMUs, write row to RingBuf ----
  if (logFlag) {
    logFlag = false;

    // Read both IMUs (direct register read, no DMP)
    readChestIMU(chestIMU, chest_data);
    readIMU(backIMU, back_data);

    // Read ECG — if FIFO overflowed, re-sync
    int32_t ch0, ch1;
    if (ecgADC.isDataReady()) {
      if (!ecgADC.readChannels(ch0, ch1)) {
        ecgADC.readChannels(ch0, ch1);  // re-sync
      }
      latest_ecg_ch0 = ch0;
      latest_ecg_ch1 = ch1;
    }

    char ts[14];
    formatTimestamp(ts, sizeof(ts));

    char row[170];
    int len = snprintf(row, sizeof(row),
      "%s,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
      ts,
      (long)latest_ecg_ch0, (long)latest_ecg_ch1,
      chest_data.ax, chest_data.ay, chest_data.az,
      chest_data.gx, chest_data.gy, chest_data.gz,
      back_data.ax,  back_data.ay,  back_data.az,
      back_data.gx,  back_data.gy,  back_data.gz
    );
    rb.write(row, len);
  }

  // ---- 2. Drain RingBuf to SD in 512-byte sectors ----
  // Periodic sync so power loss only loses ~60s of data
  static uint32_t last_sync = 0;
  if ((millis() - last_sync > 60000) && !file.isBusy()) {
    rb.sync();
    file.flush();
    last_sync = millis();
  }

  // ---- 3. Check for file full ----
  if ((rb.bytesUsed() + file.curPosition()) > (LOG_FILE_SIZE - 512)) {
    stopLogging("File full");
  }
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

bool initChestIMU(ICM42670 &imu) {
  int ret;
  ret = imu.begin();
  if (ret != 0) {
    return false;
  }
  return true;
}

void configureChestIMU(ICM42670 &imu ) {
  // Accel ODR = 100 Hz and Full Scale Range = 8G
  imu.startAccel(LOG_RATE_HZ, CHEST_ACCEL_FSR);
  // Gyro ODR = 100 Hz and Full Scale Range = 2000 dps
  imu.startGyro(LOG_RATE_HZ, CHEST_GYRO_FSR);
  // Wait IMU to start
  delay(100);
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

  // Set sample rate divider for 562.5 Hz so data is always fresh
  // ICM-20948 base rate = 1125 Hz for accel/gyro
  // Divider = (1125 / desired_rate) - 1 = (1125/562.5) - 1 ≈ 1
  // Actual rate = 1125 / (1+1) = 562.5 Hz
  ICM_20948_smplrt_t smplrt;
  smplrt.a = 1;  // accel: 1125/(1+1) ≈ 562.5 Hz
  smplrt.g = 1;  // gyro:  1125/(1+1) ≈ 562.5 Hz
  imu.setSampleRate(
    (ICM_20948_Internal_Acc | ICM_20948_Internal_Gyr),
    smplrt
  );
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
void readChestIMU(ICM42670 &imu, IMUData &data) {
  inv_imu_sensor_event_t imu_event;
  imu.getDataFromRegisters(imu_event);
  data.ax = imu_event.accel[0];
  data.ay = imu_event.accel[1];
  data.az = imu_event.accel[2];
  data.gx = imu_event.gyro[0];
  data.gy = imu_event.gyro[1];
  data.gz = imu_event.gyro[2];
  }

// ========================== Logging Control ==========================

void stopLogging(const char* reason) {
  logTimer.end();  // Stop the 100 Hz timer

  // Flush all remaining RingBuf data to file
  rb.sync();
  file.truncate();  // Trim pre-allocated space to actual size
  file.close();

  // Blink slowly to indicate stopped
  while (true) {
    digitalWrite(LED_PIN, HIGH);
    delay(500);
    digitalWrite(LED_PIN, LOW);
    delay(500);
  }
}

void fatalError() {
  logFile.println(F("\n*** FATAL ERROR - System halted ***"));
  logFile.flush();
  while (true) {
    digitalWrite(LED_PIN, HIGH);
    delay(100);
    digitalWrite(LED_PIN, LOW);
    delay(100);
  }
}

void formatTimestamp(char* buf, size_t bufSize) {
  uint32_t ms = millis() % 1000;  // sub-second millis
  snprintf(buf, bufSize, "%02d:%02d:%02d.%03lu",
    hour(), minute(), second(), (unsigned long)ms);
}

time_t getTeensy3Time()
{
  return Teensy3Clock.get();
}
