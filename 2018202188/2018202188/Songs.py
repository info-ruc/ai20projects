import RPI.GPIO as GPIO
import time

Buzzer = 11

#C大调各音调的频率
cl = [0, 131, 147, 165, 175, 196, 211, 248]
cm = [0, 262, 294, 330, 350, 393, 441, 495]
ch = [0, 525, 589, 661, 700, 786, 882, 990]

song = [cm[1], cm[2], cm[3], cm[1], cm[1], cm[2], cm[3], cm[1],
        cm[3], cm[4], cm[5], cm[3], cm[4], cm[5],
        cm[5], cm[6], cm[5], cm[4], cm[3], cm[1],
        cm[5], cm[6], cm[5], cm[4], cm[3], cm[1],
        cm[2], cl[5], cm[1], cm[2], cl[5], cm[1]]

beat = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 4,
        1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2,
        2, 2, 4, 2, 2, 4]

def setup():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)		# Numbers GPIOs by physical location
    GPIO.setup(Buzzer, GPIO.OUT)	# Set pins' mode is output
    global Buzz						# Assign a global variable to replace GPIO.PWM
    Buzz = GPIO.PWM(Buzzer, 440)	# 440 is initial frequency.
    Buzz.start(50)	                # Start Buzzer pin with 50% duty ration

def loop():
	while True:
		print '\n|Playing Two Tigers...|'
		for i in range(1, len(song)):		# Play song 1
			Buzz.ChangeFrequency(song[i])	# Change the frequency along the song note
			time.sleep(beat[i] * 0.2)		# delay a note for beat * 0.5s

def destory():
	Buzz.stop()					# Stop the buzzer
	GPIO.output(Buzzer, 1)		# Set Buzzer pin to High
	GPIO.cleanup()				# Release resource

if __name__ == '__main__':		# Program start from here
	setup()
	try:
		loop()
	except KeyboardInterrupt:  	# When 'Ctrl+C' is pressed, the child program destroy() will be  executed.
		destory()