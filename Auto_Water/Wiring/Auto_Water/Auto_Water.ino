/*--------------------------------------------------------------
  Program:      Auto_Water

  Description:  Reads in the Voltage at A0 and calculates the total
                Resistance.  It then compares this resistance to the 
                soil wetness calibration and powers a pump when
                the soil is too dry.
  
  Hardware:     Arduino Mini Pro with voltage divider on A0.
                Pump on D4
                
  Software:     Developed using Arduino 1.0.6 software

  Date:         August 27, 2016
 
  Author:       Nicolas Bertagnolli www.nbertagnolli.com
--------------------------------------------------------------*/


// =======================================================================
// Define Global Variables and Initializations
// =======================================================================

#define NUM_SAMPLES_TO_AVERAGE 10  // number of analog samples to average
#define SOIL_VOLTAGE .15

int pump_pin = 4;

double sum = 0.0;                   // Summ of current sample set
int sample_counter = 0;             // Counts number of samples taken
double voltage = 0.0;               // voltage taken


void setup() {
  Serial.begin(9600);
  pinMode(pump_pin, OUTPUT);
}


void loop() {
  // Read in samples to be averaged
  while(sample_counter < NUM_SAMPLES_TO_AVERAGE) {
    sum += analogRead(A1);
    sample_counter ++;
    delay(20);
  }
  
  // Average Sum
  voltage = sum / (NUM_SAMPLES_TO_AVERAGE);
  
  //Convert to actual voltage (0 - 5 Vdc)
  voltage = (voltage / 1024) * 5.0;
  
  // Reset looping variables
  sum = 0;
  sample_counter = 0;
  
  // Write voltage to serial
  Serial.println(voltage);
  
  if (voltage < SOIL_VOLTAGE) {
    digitalWrite(pump_pin, HIGH);
    delay (5000);
    digitalWrite(pump_pin, LOW);
    delay(10000);
  } 
  
}
