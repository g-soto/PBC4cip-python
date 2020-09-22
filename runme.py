import weka.core.jvm as jvm
import weka.core.packages as packages
from weka.classifiers import Classifier,Evaluation
import os
from weka.core.converters import Loader
from weka.attribute_selection import ASSearch, ASEvaluation, AttributeSelection
from weka.filters import Filter

########SetUp########################
os.environ["WEKA_HOME"] = os.path.abspath("./weka-3-8-4") #point it to your weka instalation folder
jvm.start(packages=True, max_heap_size='6g')
#####################################

########Install######################
packages.refresh_cache()
if not packages.is_installed('discriminantAnalysis'):
    print("Installing discriminantAnalysis")
    packages.install_package('discriminantAnalysis')
if not packages.is_installed('PBC4cip'):
    print("Installing PBC4cip")
    packages.install_package(os.path.abspath('./weka_packages/PBC4cip.zip'))
######################################

#############Data#####################
loader = Loader(classname="weka.core.converters.ArffLoader")
data = loader.load_file(os.path.abspath('./universities.arff'))
data.class_is_last()
######################################

#######rub1########################
print("rub1")
options = ["-S", "1" ,"-miner", "PRFramework.Core.SupervisedClassifiers.EmergingPatterns.Miners.RandomForestMinerWithoutFiltering -bagSizePercent 100 -numFeatures -1 -numTrees 10000 -builder \"PRFramework.Core.SupervisedClassifiers.DecisionTrees.Builder.DecisionTreeBuilder -distributionEvaluator \\\"PRFramework.Core.SupervisedClassifiers.DecisionTrees.DistributionEvaluators.QuinlanGain \\\" -maxDepth -1 \\\"-minimalObjByLeaf \\\" 2 -minimalSplitGain 1.0E-30\""]
classifier = Classifier(classname="weka.classifiers.trees.PBC4cip", options = options)
classifier.build_classifier(data)
print("writing to", 'outputs/rub1.txt')
with open('outputs/rub1.txt', 'w') as output_file:
    output_file.write(str(classifier))
######################################

#####rub2#############################
print("rub2")
for nf in [5, 10, 15, 20]:
    options = ["-S", "1" ,"-miner", "PRFramework.Core.SupervisedClassifiers.EmergingPatterns.Miners.RandomForestMinerWithoutFiltering -bagSizePercent 100 -numFeatures %d -numTrees 10000 -builder \"PRFramework.Core.SupervisedClassifiers.DecisionTrees.Builder.DecisionTreeBuilder -distributionEvaluator \\\"PRFramework.Core.SupervisedClassifiers.DecisionTrees.DistributionEvaluators.QuinlanGain \\\" -maxDepth -1 \\\"-minimalObjByLeaf \\\" 2 -minimalSplitGain 1.0E-30\"" % nf]
    classifier = Classifier(classname="weka.classifiers.trees.PBC4cip", options = options)
    classifier.build_classifier(data)
    print("writin to", 'outputs/rub2-%d.txt' % nf)
    with open('outputs/rub2-%d.txt' % nf, 'w') as output_file:
        output_file.write(str(classifier))
######################################

#####rub3#############################
print("rub3")
for dt in [2, 3, 4, 5]:
    options = ["-S", "1" ,"-miner", "PRFramework.Core.SupervisedClassifiers.EmergingPatterns.Miners.RandomForestMinerWithoutFiltering -bagSizePercent 100 -numFeatures -1 -numTrees 10000 -builder \"PRFramework.Core.SupervisedClassifiers.DecisionTrees.Builder.DecisionTreeBuilder -distributionEvaluator \\\"PRFramework.Core.SupervisedClassifiers.DecisionTrees.DistributionEvaluators.QuinlanGain \\\" -maxDepth %d \\\"-minimalObjByLeaf \\\" 2 -minimalSplitGain 1.0E-30\"" % dt]
    classifier = Classifier(classname="weka.classifiers.trees.PBC4cip", options = options)
    classifier.build_classifier(data)
    print("writin to", 'outputs/rub3-%d.txt' % dt)
    with open('outputs/rub3-%d.txt' % dt, 'w') as output_file:
        output_file.write(str(classifier))
######################################

#####rub4#############################
print("rub4")
for obl in [3, 4, 5, 10]:
    options = ["-S", "1" ,"-miner", "PRFramework.Core.SupervisedClassifiers.EmergingPatterns.Miners.RandomForestMinerWithoutFiltering -bagSizePercent 100 -numFeatures -1 -numTrees 10000 -builder \"PRFramework.Core.SupervisedClassifiers.DecisionTrees.Builder.DecisionTreeBuilder -distributionEvaluator \\\"PRFramework.Core.SupervisedClassifiers.DecisionTrees.DistributionEvaluators.QuinlanGain \\\" -maxDepth -1 \\\"-minimalObjByLeaf \\\" %d -minimalSplitGain 1.0E-30\"" % obl]
    classifier = Classifier(classname="weka.classifiers.trees.PBC4cip", options = options)
    classifier.build_classifier(data)
    print("writin to", 'outputs/rub4-%d.txt' % obl)
    with open('outputs/rub4-%d.txt' % obl, 'w') as output_file:
        output_file.write(str(classifier))
######################################

#####rub5#############################
print("rub5")
for ss in [0.2, 0.4, 0.6, 0.8]:
    options =  ['-P', str(ss), '-S', '1', '-num-slots', '1', '-I', '10', '-W', 'weka.classifiers.trees.PBC4cip', '--', '-S', '1', '-miner', "PRFramework.Core.SupervisedClassifiers.EmergingPatterns.Miners.RandomForestMinerWithoutFiltering -bagSizePercent 100 -numFeatures -1 -numTrees 1000 -builder \"PRFramework.Core.SupervisedClassifiers.DecisionTrees.Builder.DecisionTreeBuilder -distributionEvaluator \\\"PRFramework.Core.SupervisedClassifiers.DecisionTrees.DistributionEvaluators.QuinlanGain \\\" -maxDepth -1 \\\"-minimalObjByLeaf \\\" 2 -minimalSplitGain 1.0E-30\""]
    classifier = Classifier(classname="weka.classifiers.meta.RandomSubSpace", options = options)
    classifier.build_classifier(data)
    print("writin to", 'outputs/rub5-%f.txt' % ss)
    with open('outputs/rub5-%f.txt' % ss, 'w') as output_file:
        output_file.write(str(classifier))
######################################

#####rub6a#############################
print("rub6a")
search = ASSearch(classname="weka.attributeSelection.BestFirst", options=["-D", "1", "-N", "5"])
evaluator = ASEvaluation(classname="weka.attributeSelection.CfsSubsetEval", options=["-P", "1", "-E", "1"])
attsel = AttributeSelection()
attsel.search(search)
attsel.evaluator(evaluator)
attsel.select_attributes(data)
attributes = (x+1 for x in attsel.selected_attributes)

remove = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=['-V', "-R",','.join(map(str,attributes))])
remove.inputformat(data)
filtered = remove.filter(data)

options = ["-S", "1" ,"-miner", "PRFramework.Core.SupervisedClassifiers.EmergingPatterns.Miners.RandomForestMinerWithoutFiltering -bagSizePercent 100 -numFeatures -1 -numTrees 1000 -builder \"PRFramework.Core.SupervisedClassifiers.DecisionTrees.Builder.DecisionTreeBuilder -distributionEvaluator \\\"PRFramework.Core.SupervisedClassifiers.DecisionTrees.DistributionEvaluators.QuinlanGain \\\" -maxDepth -1 \\\"-minimalObjByLeaf \\\" 2 -minimalSplitGain 1.0E-30\""]
classifier = Classifier(classname="weka.classifiers.trees.PBC4cip", options = options)
classifier.build_classifier(filtered)
print("writin to", 'outputs/rub6-a.txt')
with open('outputs/rub6-a.txt', 'w') as output_file:
    output_file.write(str(classifier))
######################################

#####rub6b#############################
print("rub6b")
search = ASSearch(classname="weka.attributeSelection.GreedyStepwise", options=['-T', '-1.7976931348623157E308', '-N', '-1', '-num-slots', '1'])
evaluator = ASEvaluation(classname="weka.attributeSelection.CfsSubsetEval", options=["-P", "1", "-E", "1"])
attsel = AttributeSelection()
attsel.search(search)
attsel.evaluator(evaluator)
attsel.select_attributes(data)
attributes = (x+1 for x in attsel.selected_attributes)

remove = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=['-V', "-R",','.join(map(str,attributes))])
remove.inputformat(data)
filtered = remove.filter(data)

options = ["-S", "1" ,"-miner", "PRFramework.Core.SupervisedClassifiers.EmergingPatterns.Miners.RandomForestMinerWithoutFiltering -bagSizePercent 100 -numFeatures -1 -numTrees 1000 -builder \"PRFramework.Core.SupervisedClassifiers.DecisionTrees.Builder.DecisionTreeBuilder -distributionEvaluator \\\"PRFramework.Core.SupervisedClassifiers.DecisionTrees.DistributionEvaluators.QuinlanGain \\\" -maxDepth -1 \\\"-minimalObjByLeaf \\\" 2 -minimalSplitGain 1.0E-30\""]
classifier = Classifier(classname="weka.classifiers.trees.PBC4cip", options = options)
classifier.build_classifier(filtered)
print("writin to", 'outputs/rub6-b.txt')
with open('outputs/rub6-b.txt', 'w') as output_file:
    output_file.write(str(classifier))
######################################

#####rub6c#############################
print("rub6c")
search = ASSearch(classname="weka.attributeSelection.Ranker", options=['-T', '-1.7976931348623157E308', '-N', '-1'])
evaluator = ASEvaluation(classname="weka.attributeSelection.ClassifierAttributeEval", options=['-execution-slots', '1', '-B', 'weka.classifiers.rules.ZeroR', '-F', '5', '-T', '0.01', '-R', '1', '-E', 'DEFAULT', '--'])

attsel = AttributeSelection()
attsel.search(search)
attsel.evaluator(evaluator)
attsel.select_attributes(data)
attributes = (x+1 for x in attsel.selected_attributes)

remove = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=['-V', "-R",','.join(map(str,attributes))])
remove.inputformat(data)
filtered = remove.filter(data)

options = ["-S", "1" ,"-miner", "PRFramework.Core.SupervisedClassifiers.EmergingPatterns.Miners.RandomForestMinerWithoutFiltering -bagSizePercent 100 -numFeatures -1 -numTrees 1000 -builder \"PRFramework.Core.SupervisedClassifiers.DecisionTrees.Builder.DecisionTreeBuilder -distributionEvaluator \\\"PRFramework.Core.SupervisedClassifiers.DecisionTrees.DistributionEvaluators.QuinlanGain \\\" -maxDepth -1 \\\"-minimalObjByLeaf \\\" 2 -minimalSplitGain 1.0E-30\""]
classifier = Classifier(classname="weka.classifiers.trees.PBC4cip", options = options)
classifier.build_classifier(filtered)
print("writin to", 'outputs/rub6-c.txt')
with open('outputs/rub6-c.txt', 'w') as output_file:
    output_file.write(str(classifier))
######################################

#####rub6d#############################
print("rub6d")
search = ASSearch(classname="weka.attributeSelection.Ranker", options=['-T', '-1.7976931348623157E308', '-N', '-1'])

evaluator = ASEvaluation(classname="weka.attributeSelection.ReliefFAttributeEval", options=['-M', '-1', '-D', '1', '-K', '10'])
 
attsel = AttributeSelection()
attsel.search(search)
attsel.evaluator(evaluator)
attsel.select_attributes(data)
attributes = (x+1 for x in attsel.selected_attributes)

remove = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=['-V', "-R",','.join(map(str,attributes))])
remove.inputformat(data)
filtered = remove.filter(data)

options = ["-S", "1" ,"-miner", "PRFramework.Core.SupervisedClassifiers.EmergingPatterns.Miners.RandomForestMinerWithoutFiltering -bagSizePercent 100 -numFeatures -1 -numTrees 1000 -builder \"PRFramework.Core.SupervisedClassifiers.DecisionTrees.Builder.DecisionTreeBuilder -distributionEvaluator \\\"PRFramework.Core.SupervisedClassifiers.DecisionTrees.DistributionEvaluators.QuinlanGain \\\" -maxDepth -1 \\\"-minimalObjByLeaf \\\" 2 -minimalSplitGain 1.0E-30\""]
classifier = Classifier(classname="weka.classifiers.trees.PBC4cip", options = options)
classifier.build_classifier(filtered)
print("writin to", 'outputs/rub6-d.txt')
with open('outputs/rub6-d.txt', 'w') as output_file:
    output_file.write(str(classifier))
######################################


print("finito")


jvm.stop()