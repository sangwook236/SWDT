int analogPin = 0;     // connected to analog pin 0
                       // outside leads to ground and +5V
int val = 0;           // variable to store the value read

void setup()
{
  Serial.begin(115200);  // setup serial.
}

void loop()
{
  val = analogRead(analogPin);  // read the input pin.
  Serial.println(val);  // debug value.
}

