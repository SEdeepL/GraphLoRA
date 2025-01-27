import os
import shutil
path_patch = '' #the path of datasets
mutator_pattern ={'BooleanFalseReturnValsMutator':0, 'BooleanTrueReturnValsMutator':0,
                  'EmptyObjectReturnValsMutator':0, 'NonVoidMethodCallMutator':0,
                  'NullReturnValsMutator':0, 'RemoveConditionalMutator_EQUAL_ELSE':0,
                  'RemoveConditionalMutator_EQUAL_IF':0, 'RemoveConditionalMutator_ORDER_ELSE':0,
                  'RemoveConditionalMutator_ORDER_IF':0, 'RemoveIncrementsMutator':0, 
                  'ReturningMethodCallGuardMutator':0, 'ReturnValsMutator':0}
# ipdb.set_trace()
mutator_list = []
cnt = 0
for root, dirs, files in os.walk(path_patch):
    for file in files:
        if "mutant-info.log" not in files or 'CANT_FIX' in files or 'NO_DIFF' in files:
            break
        else:
            f = open(root+'/'+'mutant-info.log','r')
            # print("***Line 33")
            con = f.readlines()
            for s in con:
                # print("***Line 36")
                if s.startswith('\tMutator'):
                    mutator = s.split(' ')[1].replace('\n','')
                    if mutator in mutator_pattern.keys():
                        mutator = mutator
                    else:
                        mutator = False
                    print(mutator)
                    break
        break
