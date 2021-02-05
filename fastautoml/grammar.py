# Grammatical Evolution

import re

from scipy.optimize import differential_evolution


    # WARNING: Experimental, do not use in production
    # TODO: build a sklearn wrapper around the model
    def GrammaticalEvolution(self):
        
        # A grammar is a dictionary keyed by non terminal symbols
        #     Each value is a list with the posible replacements
        #         Each replacement contains a list with tokens
        #
        # The grammar in use is:
        #
        #     <expression> ::= self.X_[:,<feature>] |
        #                      <number> <scale> self.X_[:,<feature>] |
        #                      self.X_[:,<feature>]) ** <exponent> |
        #                      (<expression>) <operator> (<expression>)
        #                 
        #     <operator>   ::= + | - | * | /
        #     <scale>      ::= *
        #     <number>     ::= <digit> | <digit><digit0> | | <digit><digit0><digit0>
        #     <digit>      ::= 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
        #     <digit0>     ::= 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
        #     <exponent>   ::= 2 | 3 | (1/2) | (1/3)
        #     <feature>    ::= 1 .. self.X_.shape[1]

        self.grammar = {
            "expression": [
                            ["self.X_[:,", "<feature>", "]"],
                            ["<number>", "<scale>", "self.X_[:,", "<feature>", "]"],
                            ["self.X_[:,", "<feature>", "]**", "<exponent>"],
                            ["(", "<expression>", ")", "<operator>", "(", "<expression>", ")"]
                          ],
            "operator":   ["+", "-", "*", "/"],
            "scale":      ["*"],
            "number":     [
                            ["<digit>"], 
                            ["<digit>", "<digit0>"],
                            ["<digit>", "<digit0>", "<digit0>"]
                          ],
            "digit":      ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "digit0":     ["0", "5"],
            "exponent":   ["2", "3", "(1/2)", "(1/3)"],
            "feature":    None
        }

        # Fill in features         
        self.grammar["feature"] = [str(i) for i in np.arange(0, self.X_.shape[1])]

        self.max_num_tokens  = 10 # Sufficient to cover all possible tokens from grammar
        self.max_num_derivations = self.max_num_tokens * self.max_num_tokens # TODO: Think about that

        # Use differential evolution to find the optimal model
        bounds = [(0, self.max_num_tokens)] * self.max_num_derivations
        result = differential_evolution(self._evaluate_genotype, bounds)
        
        # Retrieve model
        model = self._parse_grammar(result.x)
        
        # Compute the predicted values
        pred = eval(model)

        # Compute model string
        model_str = model.replace("self.", "")
        
        # Compute the variables in use
        viu          = np.zeros(self.X_.shape[1], dtype=int)                    
        match        = re.compile(r'self.X_\[:,(\d+)\]') 
        indices      = match.findall(model) 
        indices      = [int(i) for i in indices] 
        viu[indices] = 1

        # Compute the nescience
        nsc = self.nescience_.nescience(None, subset=viu, predictions=pred, model_string=model_str)
        
        return (nsc, model, viu)


    """
    Given a genotype (a list of integers) compute the nescience of the
    corresponding phenotype given the grammar.
    
    Return the nescience of the phenotype
    """
    def _evaluate_genotype(self, x):
                
        # Retrieve model
        model = self._parse_grammar(x)
                
        # Compute the predicted values
        try:
            pred = eval(model)
        except:
            # In case of non-evaluable model, return a nescience of 1
            return 1 
                            
        # Compute a simplified version of model string
        model_str = model.replace("self.", "")
                
        # Compute the variables in use
        viu          = np.zeros(self.X_.shape[1], dtype=int)                    
        match        = re.compile(r'self.X_\[:,(\d+)\]') 
        indices      = match.findall(model) 
        indices      = [int(i) for i in indices] 
        viu[indices] = 1
        
        # Compute the nescience
        try:
            nsc = self.nescience_.nescience(None, subset=viu, predictions=pred, model_string=model_str)
        except:
            # In case of non-computable nesciencee, return a value of 1
            return 1 
                
        return nsc


    """
    Given a genotype (a list of integers) compute the  corresponding phenotype
    given the grammar.
    
    Return a string based phenotype
    """
    def _parse_grammar(self, x):
        
        x = [int(round(i)) for i in x]
        
        phenotype = ["<expression>"]
        ind       = 0
        modified  = True
        
        # Meanwhile there are no more non-terminal symbols
        while modified:
            
            modified = False
            new_phenotype = list()
                        
            for token in phenotype:
                            
                if token[0] == '<' and token[-1] == '>':
                    
                    token     = token[1:-1]
                    new_token = self.grammar[token][x[ind] % len(self.grammar[token])]
                                        
                    if type(new_token) == str:
                        new_token = list(new_token)
                                            
                    new_phenotype = new_phenotype + new_token
                    modified = True
                    ind = ind + 1
                    ind = ind % self.max_num_derivations
                                        
                else:
                                   
                    # new_phenotype = new_phenotype + list(token)
                    new_phenotype.append(token)
                         
            phenotype = new_phenotype
                    
        model = "".join(phenotype)

        return model
