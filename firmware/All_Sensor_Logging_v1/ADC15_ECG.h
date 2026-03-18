/*
 * ADC15_ECG.h - Simple ECG Data Acquisition Library for Teensy 4.1
 * 
 * This library interfaces with the Mikroe ADC15 Click board (ADS131M02 + LTC6903)
 * for raw ECG data acquisition. Designed for Arduino IDE compatibility.
 * 
 * No digital filtering is performed - raw ADC values are provided for
 * external processing. This gives you full control over your signal processing.
 */

#ifndef ADC15_ECG_H
#define ADC15_ECG_H

#include <Arduino.h>
#include <SPI.h>

// Pin connections - modify these based on your wiring to the Teensy 4.1
// These use standard Teensy 4.1 SPI pins for best performance
#define ADC15_CS_PIN        10  // Chip Select for ADS131M02 (any digital pin)
#define ADC15_CS2_PIN       9   // Chip Select for LTC6903 (any digital pin)
#define ADC15_DRDY_PIN      22   // Data Ready (use interrupt-capable pin)
#define ADC15_RESET_PIN     3   // Hardware reset (any digital pin)

// SPI pins on Teensy 4.1 (hardware SPI for speed)
// MOSI = 11, MISO = 12, SCK = 13 (these are fixed for SPI)

// ADS131M02 Register addresses - these define where different settings are stored
#define REG_ID              0x00  // Device identification
#define REG_STATUS          0x01  // Current device status
#define REG_MODE            0x02  // Operating mode configuration
#define REG_CLOCK           0x03  // Clock settings
#define REG_GAIN1           0x04  // Channel 1 gain setting
#define REG_CFG             0x06  // General configuration
#define REG_CH0_CFG         0x09  // Channel 0 specific config
#define REG_CH1_CFG         0x0E  // Channel 1 specific config

// ADS131M02 Commands - these tell the ADC what to do
#define CMD_NULL            0x0000  // No operation
#define CMD_RESET           0x0011  // Reset the device
#define CMD_STANDBY         0x0022  // Enter standby mode
#define CMD_WAKEUP          0x0033  // Wake from standby
#define CMD_RREG            0xA000  // Read register command
#define CMD_WREG            0x6000  // Write register command

// Gain settings - determines input voltage range
// Choose based on your ECG signal amplitude after analog filtering
#define GAIN_1              0x00  // ±1.2V range (lowest gain)
#define GAIN_2              0x01  // ±600mV range
#define GAIN_4              0x02  // ±300mV range
#define GAIN_8              0x03  // ±150mV range
#define GAIN_16             0x04  // ±75mV range
#define GAIN_32             0x05  // ±37.5mV range (typical for ECG)
#define GAIN_64             0x06  // ±18.75mV range
#define GAIN_128            0x07  // ±9.375mV range (highest gain)

// Oversampling ratio - affects noise and data rate
// Higher OSR = less noise but lower max sample rate
#define OSR_128             0x00  // Fastest, more noise
#define OSR_256             0x01
#define OSR_512             0x02
#define OSR_1024            0x03  // Good balance for ECG
#define OSR_2048            0x04
#define OSR_4096            0x05  // Slowest, least noise

// Power modes - trade power consumption for performance
#define PWR_HR              0x00  // High-resolution (best quality)
#define PWR_LP              0x01  // Low-power
#define PWR_VLP             0x02  // Very low-power

class ADC15_ECG {
public:
    // Constructor
    ADC15_ECG();
    
    // Initialization - call this in setup()
    bool begin();
    
    // Configure ECG sampling
    bool configureSampling(uint16_t sampleRate, uint8_t gain);
    
    // Read data - call this when data is ready
    bool readChannels(int32_t &channel1_mV, int32_t &channel2_mV);
    
    // Check if new data is available
    bool isDataReady();
    
    // Get current configuration
    uint16_t getSampleRate() { return currentSampleRate; }
    uint8_t getGain(uint8_t channel);
    
    // Low-level register access if needed
    bool writeRegister(uint8_t address, uint16_t value);
    uint16_t readRegister(uint8_t address);
    
    // Utility
    bool reset();
    uint16_t readID();
    
private:
    // Configuration tracking
    uint16_t currentSampleRate;
    uint8_t currentGain[2];
    uint8_t wordLength;
    
    // SPI communication
    bool spiCommand(uint16_t command);
    bool spiTransferADC(uint8_t* tx, uint8_t* rx, uint8_t len);
    bool configureLTC6903(uint32_t frequency);
    
    // Data conversion
    int32_t convertRawToSigned(uint8_t* data);
    float convertToMillivolts(int32_t raw, uint8_t gain);
    
    // CRC for data integrity
    uint16_t calculateCRC(uint8_t* data, uint8_t len);
};

#endif // ADC15_ECG_H
