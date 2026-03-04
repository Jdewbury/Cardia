#include <cstddef>
/*
 * ADC15_ECG.cpp - Implementation of ECG Data Acquisition Library
 * 
 * This file implements the communication protocols and data handling
 * for interfacing with the ADS131M02 ADC and LTC6903 clock generator
 * on the Mikroe ADC15 Click board.
 */

#include "ADC15_ECG.h"

// CRC lookup table for fast CRC16-CCITT calculation
static const uint16_t crc16_table[256] = {
    0x0000, 0x1021, 0x2042, 0x3063, 0x4084, 0x50A5, 0x60C6, 0x70E7,
    0x8108, 0x9129, 0xA14A, 0xB16B, 0xC18C, 0xD1AD, 0xE1CE, 0xF1EF,
    0x1231, 0x0210, 0x3273, 0x2252, 0x52B5, 0x4294, 0x72F7, 0x62D6,
    0x9339, 0x8318, 0xB37B, 0xA35A, 0xD3BD, 0xC39C, 0xF3FF, 0xE3DE,
    0x2462, 0x3443, 0x0420, 0x1401, 0x64E6, 0x74C7, 0x44A4, 0x5485,
    0xA56A, 0xB54B, 0x8528, 0x9509, 0xE5EE, 0xF5CF, 0xC5AC, 0xD58D,
    0x3653, 0x2672, 0x1611, 0x0630, 0x76D7, 0x66F6, 0x5695, 0x46B4,
    0xB75B, 0xA77A, 0x9719, 0x8738, 0xF7DF, 0xE7FE, 0xD79D, 0xC7BC,
    0x48C4, 0x58E5, 0x6886, 0x78A7, 0x0840, 0x1861, 0x2802, 0x3823,
    0xC9CC, 0xD9ED, 0xE98E, 0xF9AF, 0x8948, 0x9969, 0xA90A, 0xB92B,
    0x5AF5, 0x4AD4, 0x7AB7, 0x6A96, 0x1A71, 0x0A50, 0x3A33, 0x2A12,
    0xDBFD, 0xCBDC, 0xFBBF, 0xEB9E, 0x9B79, 0x8B58, 0xBB3B, 0xAB1A,
    0x6CA6, 0x7C87, 0x4CE4, 0x5CC5, 0x2C22, 0x3C03, 0x0C60, 0x1C41,
    0xEDAE, 0xFD8F, 0xCDEC, 0xDDCD, 0xAD2A, 0xBD0B, 0x8D68, 0x9D49,
    0x7E97, 0x6EB6, 0x5ED5, 0x4EF4, 0x3E13, 0x2E32, 0x1E51, 0x0E70,
    0xFF9F, 0xEFBE, 0xDFDD, 0xCFFC, 0xBF1B, 0xAF3A, 0x9F59, 0x8F78,
    0x9188, 0x81A9, 0xB1CA, 0xA1EB, 0xD10C, 0xC12D, 0xF14E, 0xE16F,
    0x1080, 0x00A1, 0x30C2, 0x20E3, 0x5004, 0x4025, 0x7046, 0x6067,
    0x83B9, 0x9398, 0xA3FB, 0xB3DA, 0xC33D, 0xD31C, 0xE37F, 0xF35E,
    0x02B1, 0x1290, 0x22F3, 0x32D2, 0x4235, 0x5214, 0x6277, 0x7256,
    0xB5EA, 0xA5CB, 0x95A8, 0x8589, 0xF56E, 0xE54F, 0xD52C, 0xC50D,
    0x34E2, 0x24C3, 0x14A0, 0x0481, 0x7466, 0x6447, 0x5424, 0x4405,
    0xA7DB, 0xB7FA, 0x8799, 0x97B8, 0xE75F, 0xF77E, 0xC71D, 0xD73C,
    0x26D3, 0x36F2, 0x0691, 0x16B0, 0x6657, 0x7676, 0x4615, 0x5634,
    0xD94C, 0xC96D, 0xF90E, 0xE92F, 0x99C8, 0x89E9, 0xB98A, 0xA9AB,
    0x5844, 0x4865, 0x7806, 0x6827, 0x18C0, 0x08E1, 0x3882, 0x28A3,
    0xCB7D, 0xDB5C, 0xEB3F, 0xFB1E, 0x8BF9, 0x9BD8, 0xABBB, 0xBB9A,
    0x4A75, 0x5A54, 0x6A37, 0x7A16, 0x0AF1, 0x1AD0, 0x2AB3, 0x3A92,
    0xFD2E, 0xED0F, 0xDD6C, 0xCD4D, 0xBDAA, 0xAD8B, 0x9DE8, 0x8DC9,
    0x7C26, 0x6C07, 0x5C64, 0x4C45, 0x3CA2, 0x2C83, 0x1CE0, 0x0CC1,
    0xEF1F, 0xFF3E, 0xCF5D, 0xDF7C, 0xAF9B, 0xBFBA, 0x8FD9, 0x9FF8,
    0x6E17, 0x7E36, 0x4E55, 0x5E74, 0x2E93, 0x3EB2, 0x0ED1, 0x1EF0
};

// Constructor - initialize member variables to safe defaults
ADC15_ECG::ADC15_ECG() {
    currentSampleRate = 600;  // Default 500 Hz for ECG
    currentGain[0] = GAIN_32; // Typical ECG gain
    currentGain[1] = GAIN_32;
    wordLength = 3;            // 24-bit mode
}
 
// Initialize the ADC system - this sets up all hardware connections
bool ADC15_ECG::begin() {
    // Configure the pin modes for control signals
    pinMode(ADC15_CS_PIN, OUTPUT);
    pinMode(ADC15_CS2_PIN, OUTPUT);
    pinMode(ADC15_RESET_PIN, OUTPUT);
    pinMode(ADC15_DRDY_PIN, INPUT_PULLUP);  // Use pullup for DRDY
    
    // Keep both chips deselected initially
    digitalWrite(ADC15_CS_PIN, HIGH);
    digitalWrite(ADC15_CS2_PIN, HIGH);
    digitalWrite(ADC15_RESET_PIN, HIGH);
    
    // Initialize the SPI bus
    SPI.begin();

    // Perform hardware reset to ensure clean state
    Serial.println("Performing hardware reset...");
    digitalWrite(ADC15_RESET_PIN, LOW);
    delay(50);  // Hold reset low for 50ms (longer for safety)
    digitalWrite(ADC15_RESET_PIN, HIGH);
    delay(200); // Wait longer for device to fully initialize
    
    // Configure the LTC6903 to provide the master clock
    // The ADS131M02 REQUIRES a continuous clock to operate
    Serial.println("Configuring LTC6903 clock generator...");
    if (!configureLTC6903(8192000)) {
        Serial.println("ERROR: Failed to configure LTC6903!");
        return false;
    }
    
    delay(50); // Let the clock stabilize longer
    
    // First, let's try to wake up the device and unlock registers
    Serial.println("Waking up ADC...");
    spiCommand(CMD_WAKEUP);
    delay(10);
    
    // Unlock the registers
    spiCommand(0x0655);  // UNLOCK command
    delay(10);
    
    // Send NULL command to establish communication
    spiCommand(CMD_NULL);
    delay(10);
    
    // Send a reset command to the ADC
    Serial.println("Sending software reset...");
    if (!reset()) {
        Serial.println("WARNING: Reset command may have failed");
    }
    delay(100);  // Wait after reset
    
    // Try reading the device ID multiple times
    Serial.println("Reading device ID...");
    uint16_t deviceID = 0;
    for (int attempt = 0; attempt < 3; attempt++) {
        deviceID = readID();
        Serial.print("Attempt ");
        Serial.print(attempt + 1);
        Serial.print(" - Device ID: 0x");
        Serial.println(deviceID, HEX);
        
        if ((deviceID & 0xFF00) == 0x2200) { // ADS131M02 ID should be 0x22xx
            Serial.println("Device ID verified!");
            break;
        }
        delay(50);
    }
    
    if ((deviceID & 0xFF00) != 0x2200) {
        Serial.println("ERROR: Invalid device ID!");
        return false;
    }
    
    // Now configure the device with proper 24-bit mode
    Serial.println("Configuring MODE register...");
    
    // MODE register: Set for 24-bit, High-res, OSR=1024
    // For 24-bit mode, bits [15:14] should be 01
    uint16_t modeValue = 0x4100; // 24-bit words, OSR=1024, HR mode
    if (!writeRegister(REG_MODE, modeValue)) {
        Serial.println("ERROR: Failed to write MODE register!");
        return false;
    }
    delay(10);
    
    // Configure the CLOCK register
    Serial.println("Configuring CLOCK register...");
    // Enable both channels, set appropriate clock divider
    uint16_t clockValue = 0x031A; // Enable CH0 and CH1, proper divider
    if (!writeRegister(REG_CLOCK, clockValue)) {
        Serial.println("ERROR: Failed to write CLOCK register!");
        return false;
    }
    delay(10);
    
    // Set initial gain for both channels
    Serial.println("Setting channel gains...");
    uint16_t gainValue = ((currentGain[1] & 0x07) << 4) | (currentGain[0] & 0x07);
    if (!writeRegister(REG_GAIN1, gainValue)) {
        Serial.println("ERROR: Failed to set channel gains!");
        return false;
    }
    delay(10);
    
    // Configure general settings
    Serial.println("Configuring general settings...");
    // Enable CRC, enable global chop for better performance
    uint16_t cfgValue = 0x0E00; // CRC enabled, GC_EN, proper config
    if (!writeRegister(REG_CFG, cfgValue)) {
        Serial.println("ERROR: Failed to write CFG register!");
        return false;
    }
    
    Serial.println("ADC initialization complete!");
    return true;
}

// Configure sampling parameters for ECG acquisition
bool ADC15_ECG::configureSampling(uint16_t sampleRate, uint8_t gain) {
    // Update the master clock frequency based on desired sample rate
    // The LTC6903 generates the ADC master clock
    // Clock frequency = sampleRate * OSR * 2 (approximately)
    uint32_t targetClock = sampleRate * 8192 * 2; // Using OSR=1024
    
    if (!configureLTC6903(targetClock)) {
        return false;
    }
    
    currentSampleRate = sampleRate;
    
    // Set the gain for both channels (usually the same for ECG)
    currentGain[0] = gain;
    currentGain[1] = gain;
    
    uint16_t gainValue = ((gain & 0x07) << 4) | (gain & 0x07);
    if (!writeRegister(REG_GAIN1, gainValue)) {
        return false;
    }
    
    return true;
}

// Read data from both ADC channels
bool ADC15_ECG::readChannels(int32_t &channel1_raw, int32_t &channel2_raw) {
    // In 24-bit mode, each word is 3 bytes
    // Frame structure: [STATUS(3)] [CH0_DATA(3)] [CH1_DATA(3)] [CRC(3)]
    // Total: 12 bytes for a complete data frame
    
    uint8_t txData[12] = {0};  // All zeros for NULL command
    uint8_t rxData[12] = {0};  // Receive buffer
    
    // Send NULL command to read the latest conversion data
    if (!spiTransferADC(txData, rxData, 12)) {
        Serial.println("ERROR: SPI transfer failed!");
        return false;
    }
    
    // Debug: Print received bytes
    if (false) {  // Set to true for debugging
        Serial.print("Received bytes: ");
        for(int i = 0; i < 12; i++) {
            if(rxData[i] < 0x10) Serial.print("0");
            Serial.print(rxData[i], HEX);
            Serial.print(" ");
        }
        Serial.println();
    }
    
    // Check if we got valid data (not all zeros or all FFs)
    bool allZero = true;
    bool allFF = true;
    for(int i = 0; i < 12; i++) {
        if(rxData[i] != 0x00) allZero = false;
        if(rxData[i] != 0xFF) allFF = false;
    }
    
    if(allZero || allFF) {
        Serial.println("WARNING: Invalid data pattern detected");
        return false;
    }
    
    // Extract and verify CRC (bytes 9-11)
    uint16_t receivedCRC = (rxData[9] << 8) | rxData[10];
    uint16_t calculatedCRC = calculateCRC(rxData, 9);
    
    if (receivedCRC != calculatedCRC) {
        Serial.print("CRC mismatch! Received: 0x");
        Serial.print(receivedCRC, HEX);
        Serial.print(" Calculated: 0x");
        Serial.println(calculatedCRC, HEX);
        
        // For now, continue even with CRC error for debugging
        return false;
    }
    
    // Extract status word (bytes 0-2)
    uint32_t status = ((uint32_t)rxData[0] << 16) | ((uint32_t)rxData[1] << 8) | rxData[2];
    
    // Extract channel 0 data (bytes 3-5)
    int32_t raw_ch0 = convertRawToSigned(&rxData[3]);
    
    // Extract channel 1 data (bytes 6-8)
    int32_t raw_ch1 = convertRawToSigned(&rxData[6]);
    
    // Debug: Print raw values
    if (false) {  // Set to true for debugging
        Serial.print("Status: 0x");
        Serial.print(status, HEX);
        Serial.print(" CH0 raw: ");
        Serial.print(raw_ch0);
        Serial.print(" CH1 raw: ");
        Serial.println(raw_ch1);
    }

    channel1_raw = raw_ch0;
    channel2_raw = raw_ch1;
    
    return true;
}

// Check if new data is ready to be read
bool ADC15_ECG::isDataReady() {
    // The DRDY pin goes low when new data is available
    return (digitalRead(ADC15_DRDY_PIN) == LOW);
}

// Get the current gain setting for a channel
uint8_t ADC15_ECG::getGain(uint8_t channel) {
    if (channel > 1) return 0;
    return currentGain[channel];
}

// Write to a register in the ADS131M02
bool ADC15_ECG::writeRegister(uint8_t address, uint16_t value) {
    // For 24-bit word mode, we need to send frames of 3 bytes each
    // Frame 1: WREG command
    // Frame 2: Register data
    // Frame 3: CRC (optional, but we'll include it)
    
    uint8_t txData[9] = {0}; // 3 words × 3 bytes = 9 bytes
    uint8_t rxData[9] = {0};
    
    // Build WREG command (first 24-bit word)
    uint16_t command = CMD_WREG | ((address & 0x3F) << 7);
    txData[0] = (command >> 8) & 0xFF;  // MSB of command
    txData[1] = command & 0xFF;         // LSB of command
    txData[2] = 0x00;                    // Padding byte for 24-bit word
    
    // Add register data (second 24-bit word)
    txData[3] = (value >> 8) & 0xFF;    // MSB of data
    txData[4] = value & 0xFF;           // LSB of data
    txData[5] = 0x00;                    // Padding byte
    
    // Calculate and add CRC (third 24-bit word)
    uint16_t crc = calculateCRC(txData, 6);
    txData[6] = (crc >> 8) & 0xFF;      // MSB of CRC
    txData[7] = crc & 0xFF;              // LSB of CRC
    txData[8] = 0x00;                    // Padding byte
    
    // Send the complete frame
    bool result = spiTransferADC(txData, rxData, 9);
    
    // The response to WREG comes in the next frame
    // Send NULL command to get the acknowledgment
    delay(1);
    uint8_t nullTx[3] = {0, 0, 0};
    uint8_t nullRx[3] = {0, 0, 0};
    spiTransferADC(nullTx, nullRx, 3);
    
    return result;
}

// Read from a register in the ADS131M02
uint16_t ADC15_ECG::readRegister(uint8_t address) {
    uint8_t txData[6] = {0};
    uint8_t rxData[9] = {0};
    
    // Build RREG command (first 24-bit word)
    uint16_t command = CMD_RREG | ((address & 0x3F) << 7);
    txData[0] = (command >> 8) & 0xFF;
    txData[1] = command & 0xFF;
    txData[2] = 0x00;  // Padding for 24-bit alignment

    Serial.print("Calling command: "); 
    Serial.print(txData[0], HEX);
    Serial.print(txData[1], HEX);
    Serial.print(txData[2], HEX);
    Serial.println();
    
    //Add CRC for command (second 24-bit word)
    uint16_t crc = calculateCRC(txData, 3);
    txData[3] = (crc >> 8) & 0xFF;
    txData[4] = crc & 0xFF;
    txData[5] = 0x00;  // Padding
    
    // Send the read command
    if (!spiTransferADC(txData, rxData, 3)) {
        Serial.println("ERROR: Failed to send RREG command");
        return 0;
    }
    
    // The response comes in the NEXT frame
    // Send NULL command to clock out the response
    delay(1);  // Small delay between frames
    
    memset(txData, 0, 6);
    memset(rxData, 0, 9);
    
    // Send NULL to get the register data
    if (!spiTransferADC(txData, rxData, 9)) {
        Serial.println("ERROR: Failed to read RREG response");
        return 0;
    }
    
    // Debug: Print what we received
    Serial.print("RREG Response bytes: ");
    for(int i = 0; i < 9; i++) {
        Serial.print("0x");
        if(rxData[i] < 0x10) Serial.print("0");
        Serial.print(rxData[i], HEX);
        Serial.print(" ");
    }
    Serial.println();
    
    // The register value should be in the first 24-bit word (bytes 0-2)
    uint16_t regValue = (rxData[0]<<8) | rxData[1];
    
    return regValue;
}

// Reset the ADC to default state
bool ADC15_ECG::reset() {
    return spiCommand(CMD_RESET);
}

// Read the device ID register
uint16_t ADC15_ECG::readID() {
    return readRegister(REG_ID);
}

// Send a simple command to the ADC
bool ADC15_ECG::spiCommand(uint16_t command) {
    uint8_t txData[3];
    uint8_t rxData[3];
    
    // Commands are sent as 24-bit words with the command in the upper 16 bits
    txData[0] = (command >> 8) & 0xFF;
    txData[1] = command & 0xFF;
    txData[2] = 0;
    
    return spiTransferADC(txData, rxData, 3);
}

// Perform SPI transfer with the ADC
bool ADC15_ECG::spiTransferADC(uint8_t* tx, uint8_t* rx, uint8_t len) {
    // Configure SPI for ADS131M02: Mode 1, MSB first
    // Using slower speed for reliability during initialization
    SPI.beginTransaction(SPISettings(4000000, MSBFIRST, SPI_MODE1));
    
    digitalWrite(ADC15_CS_PIN, LOW);  // Select the ADC
    delayMicroseconds(1);  // Small setup delay
    
    // The ADS131M02 expects continuous clocking
    // We need to ensure proper 24-bit alignment
    for (uint8_t i = 0; i < len; i++) {
        rx[i] = SPI.transfer(tx[i]);
    }
    
    delayMicroseconds(1);  // Small hold delay
    digitalWrite(ADC15_CS_PIN, HIGH); // Deselect the ADC
    SPI.endTransaction();
    
    // Small delay between transactions
    delayMicroseconds(10);
    
    return true;
}

// Configure the LTC6903 clock generator
bool ADC15_ECG::configureLTC6903(uint32_t frequency) {
    // The LTC6903 uses a simple formula to set frequency:
    // frequency = 2078 / (2 - DAC/1024) * 2^OCT
    // We need to solve for OCT and DAC values
    
    // Calculate OCT (octave) - essentially finding the right power of 2
    uint8_t oct = 0;
    uint32_t f_test = frequency;
    while (f_test > 2039 && oct < 15) {
        f_test >>= 1;
        oct++;
    }
    
    // Calculate DAC value for fine frequency adjustment
    float dac_float = 2048.0 - (2078.0 * pow(2, 10 + oct) / frequency);
    uint16_t dac = (uint16_t)round(dac_float);
    
    // Limit DAC to valid range
    if (dac > 1023) dac = 1023;
    
    // Build the 16-bit configuration word for LTC6903
    // Bits [15:12] = OCT (4 bits)
    // Bits [11:2]  = DAC (10 bits)  
    // Bits [1:0]   = CNF (2 bits, set to 00 for normal operation)
    uint16_t configWord = ((oct & 0x0F) << 12) | ((dac & 0x3FF) << 2) | 0x00;
    
    // Send configuration to LTC6903 via SPI
    // LTC6903 uses Mode 0 and expects 16-bit transfer
    SPI.beginTransaction(SPISettings(10000000, MSBFIRST, SPI_MODE0));
    
    digitalWrite(ADC15_CS2_PIN, LOW);  // Select the LTC6903
    
    SPI.transfer16(configWord);
    
    digitalWrite(ADC15_CS2_PIN, HIGH); // Deselect the LTC6903
    SPI.endTransaction();
    
    return true;
}

// Convert raw 24-bit ADC data to signed 32-bit integer
int32_t ADC15_ECG::convertRawToSigned(uint8_t* data) {
    // ADC data is 24-bit two's complement
    int32_t value = ((int32_t)data[0] << 16) | ((int32_t)data[1] << 8) | data[2];
    
    // Sign extend from 24-bit to 32-bit
    if (value & 0x800000) {  // If sign bit is set
        value |= 0xFF000000;  // Extend the sign
    }
    
    return value;
}

// Convert raw ADC value to millivolts based on gain setting
float ADC15_ECG::convertToMillivolts(int32_t raw, uint8_t gain) {
    // Reference voltage is 1.2V internal reference
    const float vref = 1200.0; // in millivolts
    
    // Full scale range is ±VREF/gain
    float gainMultiplier = 1.0;
    switch (gain) {
        case GAIN_1:   gainMultiplier = 1.0;   break;
        case GAIN_2:   gainMultiplier = 2.0;   break;
        case GAIN_4:   gainMultiplier = 4.0;   break;
        case GAIN_8:   gainMultiplier = 8.0;   break;
        case GAIN_16:  gainMultiplier = 16.0;  break;
        case GAIN_32:  gainMultiplier = 32.0;  break;
        case GAIN_64:  gainMultiplier = 64.0;  break;
        case GAIN_128: gainMultiplier = 128.0; break;
    }
    
    // ADC resolution is 24 bits
    const float resolution = 8388608.0; // 2^23 (since it's signed)
    
    // Calculate voltage in millivolts
    float voltage_mV = (raw / resolution) * (vref/gainMultiplier);
    
    return voltage_mV;
}

// Calculate CRC16-CCITT for data integrity checking
uint16_t ADC15_ECG::calculateCRC(uint8_t* data, uint8_t len) {
    uint16_t crc = 0xFFFF;  // Initial value for CRC16-CCITT
    
    for (uint8_t i = 0; i < len; i++) {
        uint8_t tableIndex = ((crc >> 8) ^ data[i]) & 0xFF;
        crc = ((crc << 8) ^ crc16_table[tableIndex]) & 0xFFFF;
    }
    
    return crc;
}
