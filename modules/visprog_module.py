class VisProgModule():
    
    def __init__(self):
        """ Load a trained model, move it to gpu, etc. """
        pass

    def html(self, inputs: list, output: Any):
        """ Return an html string visualizing step I/O

        Parameters
        ----------
        inputs : list
            I am guessing inputs to the module?
            
        output : Any
            I am guessing outputs from the module?
            
        """
        pass

    def parse(self, step: str):
        """ Parse step and return list of input values/variable names 
            and output variable name. 

        Parameters
        ----------
        step : str

        Returns
        -------
        inputs : ?

        input_var_names : ?

        output_var_name : ?
            
        """
        pass

    def perform_module_function(self, inputs):
        """ NOTE: I added this for us. The idea is we can implement
            this, parse, and html, and just call execute in a loop on 
            a list of VisProgModules.

        Parameters
        ----------
        inputs : ? 
            Likely just images for our task... this doesn't look to be typed
            in the original Visprog paper... so will need to take some liberties here
            and just use a torch tensor or image... can also leave it untyped
        """

        pass

    def execute(self, step: str, state: dict):
        inputs, input_var_names, output_var_name = self.parse(step)
        
        # Get values of input variables form state
        for var_name in input_var_names:
            inputs.append(state[var_name])

        # Perform computation using the loaded module
        output = peform_module_function(inputs)

        # Update state
        state[output_var_name] = output

        step_html = self.html(inputs, output)

        return output, step_html


