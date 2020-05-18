import os, sys
import unittest

#import importlib

if __name__ == "__main__":
    #include path:
    print("PWD", os.getcwd())
    raw_path = os.getcwd()
    include_path = raw_path.split("/Ensembler")[0]
    sys.path.append(include_path)
    print("sys path append:",include_path)

    #FILE MANAGMENT
    test_root_dir = raw_path #.replace("Ensembler", "")
    print("TEST ROOT DIR: " + test_root_dir)

    ##gather all test_files
    test_files = []
    exceptions_file = ["ensemble"]

    for dir in os.walk(test_root_dir):
        test_files.extend([dir[0]+"/"+path for path in dir[2] if(path.startswith("test") and path.endswith(".py") and not path in exceptions_file)])
    if(len(test_files) == 0):
        print("Could not find any test in : ", test_root_dir)
        exit(1)

    ##get module import paths - there should be a function for that around
    modules = []
    for test_module in test_files:
        module_name =  test_module.replace(os.path.dirname(test_root_dir), "").replace("/", ".").replace("\\", ".").replace(".py", "")
        if(module_name.startswith(".")): module_name = module_name[1:]
        modules.append(module_name)

    #LOAD TESTS
    print("LOAD TESTS")
    suite = unittest.TestSuite()
    test_loader = unittest.TestLoader()
    first = True

    for test_module in modules:
        print("Importing:\t", test_module)
        
        if("conveyor" in test_module):
            continue

        imported_test_module = __import__(test_module, globals(), locals(), ['suite'])
        
        if(first):
            suite = test_loader.loadTestsFromModule(imported_test_module)
            first = False
        else:
            tmp = test_loader.loadTestsFromModule(imported_test_module)
            suite.addTest(tmp)

    #RUN TESTS
    print("RUN TESTS")
    try:
        print("TEST SUIT TESTS: ", suite.countTestCases())
        test_runner = unittest.TextTestRunner(verbosity=5)
        test_result = test_runner.run(suite)

        print("\nSUMMARY: ")
        print("\ttests run:\t", test_result.testsRun)
        if(len(test_result.failures)>0 or len(test_result.errors)>0 ):
            print(test_result.failures)

            print("\ttest Failures: " + str(len(test_result.failures)) + " \n\t\t",
                  "\n\t\t".join(map(lambda x: str(x[0]) + "\n\t\t\t\t" + "\n\t\t\t\t".join(x[1].split("\n")), test_result.failures)))
            print("\ttest Errors: " + str(len(test_result.errors)) + " \n\t\t",
                  "\n\t\t".join(map(lambda x: str(x[0]) + "\n\t\t\t\t" + "\n\t\t\t\t".join(x[1].split("\n")) + "\n", test_result.errors)))
            print()
            raise Exception("")
        else:
            print("\ttest Failures:\tNone")
            print("\ttest Errors:\tNone")
            print()
        exit(0)
    except Exception as err:
        print("Test did not finish successfully!\n\t"+"\n\t".join(err.args))
        exit(1)
