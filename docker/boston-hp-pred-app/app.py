import numpy as np
import joblib
import ast

model = joblib.load("model")

def get_prediction(mdl_inps):
    
    X = np.array([mdl_inps.get("CRIM",0),mdl_inps.get("ZN",0),mdl_inps.get("INDUS",0),
              mdl_inps.get("CHAS",0),mdl_inps.get("NOX",0),mdl_inps.get("RM",0),
              mdl_inps.get("AGE",0),mdl_inps.get("DIS",0),mdl_inps.get("RAD",0),
              mdl_inps.get("TAX",0),mdl_inps.get("PTRATIO",0),mdl_inps.get("B",0),
              mdl_inps.get("LSTAT",0)])
    
    return f"price predicted is : {model.predict(np.reshape(X,(1,-1)))[0]}"


if __name__ == "__main__":
    print("enter the inputs in the following order :")
    print("""
          here is a sample input: 
          {"CRIM":0.03237,"ZN":0.0,"INDUS":2.18,"CHAS":0.0,"NOX":0.458,
            "RM":6.998,"AGE":45.8,"DIS":6.0622,"RAD":3.0,"TAX":222.0,
            "PTRATIO":18.7,"B":394.63,"LSTAT":2.94}
          """)
    
    mdl_inps = input("your inputs: ")
    result = get_prediction(ast.literal_eval(mdl_inps))
    print(result)  