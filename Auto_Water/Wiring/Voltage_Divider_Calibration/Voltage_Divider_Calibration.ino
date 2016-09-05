/*--------------------------------------------------------------
  Program:      Voltage_Divider_Calibration

  Description:  Reads in the Voltage at A0 and calculates the total
                Resistance.  It then prints this Value to Serial every
                second.
  
  Hardware:     Arduino Mini Pro with voltage divider on A0.
                
  Software:     Developed using Arduino 1.0.6 software

  Date:         August 27, 2016
 
  Author:       Nicolas Bertagnolli www.nbertagnolli.com
--------------------------------------------------------------*/


// =======================================================================
// Define Global Variables and Initializations
// =======================================================================

#define NUM_SAMPLES_TO_AVERAGE 10  // number of analog samples to average


double sum = 0.0;                   // Summ of current sample set
int sample_counter = 0;             // Counts number of samples taken
double voltage = 0.0;               // voltage taken


void setup() {
  Serial.begin(9600);
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
//  Serial.print('V:  ');
  Serial.println(voltage);
  delay(100);  // Wait 1 second
}
