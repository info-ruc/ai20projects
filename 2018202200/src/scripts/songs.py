import RPi.GPIO as GPIO
import time

Buzzer = 11

cl = [1, 131, 147, 165, 175, 196, 211, 248]
cm = [1, 262, 294, 330, 350, 393, 441, 495]
ch = [1, 525, 589, 661, 700, 786, 882, 990]

song = [ch[3],ch[3],ch[4],ch[5],ch[5],ch[4],ch[3],ch[2],
        ch[1],ch[1],ch[2],ch[3],ch[3],ch[2],ch[0],ch[2],
        ch[3],ch[3],ch[4],ch[5],ch[5],ch[4],ch[3],ch[2],
        ch[1],ch[1],ch[2],ch[3],ch[2],ch[1],ch[0],ch[1],
        ch[2],ch[2],ch[3],ch[1],ch[2],ch[3],ch[4],ch[3],ch[1],
        ch[2],ch[3],ch[4],ch[3],ch[2],ch[1],ch[2],cm[5],ch[3],
        ch[3],ch[3],ch[4],ch[5],ch[5],ch[4],ch[3],ch[2],
        ch[1],ch[1],ch[2],ch[3],ch[2],ch[1],ch[0],ch[1]]

beat = [4,4,4,4,4,4,4,4,4,4,4,4,7,1,0.1,8,
        4,4,4,4,4,4,4,4,4,4,4,4,7,1,0.1,8,
        4,4,4,4,4,2,2,4,4,4,2,2,4,4,4,4,4,4,
        4,4,4,4,4,4,4,4,4,4,4,4,7,1,0.1,8]

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
		for i in range(len(song)):		# Play song 1
			Buzz.ChangeFrequency(song[i])	# Change the frequency along the song note
			time.sleep(beat[i] * 0.1)		# delay a note for beat * 0.5s

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