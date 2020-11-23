import skfuzzy.control as ctrl

def createContorller():
    # Fuzzy membership centers [-2 -1 0 1 2]
    

    # Create the three fuzzy variables - two inputs, one output
    universe = np.linspace(-2, 2, 5)
    error = ctrl.Antecedent(universe, 'error')
    names = ['nb', 'ns', 'z', 'ps', 'pb']
    error.automf(names=names)

    delta = ctrl.Antecedent(universe, 'delta')
    delta.automf(names=names)

    output = ctrl.Consequent(universe, 'output')
    output.automf(names=names)

    # Name fuzzy variables
    

    # Rule 0: if the error is big negative and changing in the wrong direction, then output is big positive
    rule0 = ctrl.Rule(antecedent=((error['nb'] & delta['nb']) |
                                  (error['ns'] & delta['nb']) |
                                  (error['nb'] & delta['ns']) |
                                  (error['z'] & delta['nb']) |
                                  (error['nb'] & delta['z'])),
                      consequent=output['pb'], label='rule pb')

    # Rule 1: For somewhat more favourable combinations the output is small positive
    rule1 = ctrl.Rule(antecedent=((error['ns'] & delta['ns']) |
                                  (error['ns'] & delta['z']) |
                                  (error['nb'] & delta['ps']) |
                                  (error['ps'] & delta['nb']) |
                                  (error['z'] & delta['ns'])),
                      consequent=output['ps'], label='rule ps')

    # Rule 2: For good errors and good changes, the output is zero
    rule2 = ctrl.Rule(antecedent=((error['z'] & delta['z']) |
                                  (error['ns'] & delta['ps']) |
                                  (error['nb'] & delta['pb']) |
                                  (error['ps'] & delta['ns']) |
                                  (error['pb'] & delta['nb'])),
                      consequent=output['z'], label='rule z')

    # Rule 3: For somewhat less favourable combinations the output is small negative
    rule3 = ctrl.Rule(antecedent=((error['ps'] & delta['ps']) |
                                  (error['ps'] & delta['z']) |
                                  (error['pb'] & delta['ns']) |
                                  (error['ns'] & delta['pb']) |
                                  (error['z'] & delta['ps'])),
                      consequent=output['ns'], label='rule ns')

    # Rule 4: if the error is big positive and changing in the wrong direction, then output is big negative
    rule4 = ctrl.Rule(antecedent=((error['pb'] & delta['pb']) |
                                  (error['ps'] & delta['pb']) |
                                  (error['pb'] & delta['ps']) |
                                  (error['z'] & delta['pb']) |
                                  (error['pb'] & delta['z'])),
                      consequent=output['nb'], label='rule nb')

    # Create fuzzy controller
    system = ctrl.ControlSystem(rules=[rule0, rule1, rule2, rule3, rule4])
    controller = ctrl.ControlSystemSimulation(system)

    # Return fuzzy simulator
    return controller
