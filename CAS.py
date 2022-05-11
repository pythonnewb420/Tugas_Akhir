#actuator action
    motor.ChangeDutyCycle(cdc_motor)
    servo.ChangeDutyCycle(7.5) #neutral position for servo
    if distance < 120: #conditional for collision avoidance based on blocking area method
        servo.ChangeDutyCycle(cdc_servo)
        motor.ChangeDutyCycle(cdc_motor)
        time.sleep(5)
        break
    else:
        continue