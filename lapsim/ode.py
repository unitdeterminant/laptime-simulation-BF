def def_ode(car, car_dot):
    car_dot.position += car.velocity
    car_dot.velocity += car.acceleration
    car_dot.time += 1


def Drag(scale=8e-4):
    def drag(car, car_dot):
        car_dot.velocity -= scale * car.velocity ** 2

    return drag


def OdeList(ode_list):
    def ode(car, car_dot):
        for fn in ode_list:
            fn(car, car_dot)

    return ode


def CurveLoss(track, max_scale=1):
    def curve_loss(car, car_dot):
        curvature = abs(track.curvature(car.position))

        if curvature > 0:
            max_velocity = max_scale / curvature

            if car.velocity > max_velocity:
                car_dot.loss += 1
    
    return curve_loss
