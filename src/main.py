from CaptchaCracker import CaptchaCracker

if __name__== '__main__':
    testFile = "./resources/test/dptvkk.wav"
    cracker = CaptchaCracker()
    captcha = cracker.audioToText(testFile)
    print("Decoded captcha: " + captcha)