from rdkit import Chem

from guacamol.common_scoring_functions import TanimotoScoringFunction, RdkitScoringFunction, CNS_MPO_ScoringFunction, \
    IsomerScoringFunction, SMARTSScoringFunction
from guacamol.distribution_learning_benchmark import DistributionLearningBenchmark, NoveltyBenchmark, KLDivBenchmark
from guacamol.goal_directed_benchmark import GoalDirectedBenchmark
from guacamol.goal_directed_score_contributions import uniform_specification
from guacamol.score_modifier import MinGaussianModifier, MaxGaussianModifier, ClippedScoreModifier, GaussianModifier
from guacamol.scoring_function import ArithmeticMeanScoringFunction, GeometricMeanScoringFunction, ScoringFunction
from guacamol.utils.descriptors import num_rotatable_bonds, num_aromatic_rings, logP, qed, tpsa, bertz, mol_weight, \
    AtomCounter, num_rings


def isomers_c11h24(mean_function='geometric') -> GoalDirectedBenchmark:
    """
    Benchmark to try and get all C11H24 molecules there are.
    There should be 159 if one ignores stereochemistry.

    Args:
        mean_function: 'arithmetic' or 'geometric'
    """
    benchmark_name = f'isomers_c11h24 (mean_function: {mean_function})'

    objective = IsomerScoringFunction(id=benchmark_name, molecular_formula='C11H24', mean_function=mean_function)

    specification = uniform_specification(159)

    return GoalDirectedBenchmark(name='C11H24',
                                 objective=objective,
                                 contribution_specification=specification)


def isomers_c7h8n2o2(mean_function='geometric') -> GoalDirectedBenchmark:
    """
    Benchmark to try and get 100 isomers for C7H8N2O2.

    Args:
        mean_function: 'arithmetic' or 'geometric'
    """
    benchmark_name = f'isomers_c7h8n2o2 (mean_function: {mean_function})'

    objective = IsomerScoringFunction(id=benchmark_name, molecular_formula='C7H8N2O2', mean_function=mean_function)

    specification = uniform_specification(100)

    return GoalDirectedBenchmark(name='C7H8N2O2',
                                 objective=objective,
                                 contribution_specification=specification)


def isomers_c9h10n2o2pf2cl(mean_function='geometric', n_samples=250) -> GoalDirectedBenchmark:
    """
    Benchmark to try and get 100 isomers for C9H10N2O2PF2Cl.

    Args:
        mean_function: 'arithmetic' or 'geometric'
    """
    benchmark_name = f'isomers_c9h10n2o2pf2cl (mean_function: {mean_function})'

    objective = IsomerScoringFunction(id=benchmark_name, molecular_formula='C9H10N2O2PF2Cl', mean_function=mean_function)

    specification = uniform_specification(n_samples)

    return GoalDirectedBenchmark(name='C9H10N2O2PF2Cl',
                                 objective=objective,
                                 contribution_specification=specification)


def hard_cobimetinib(max_logP=5.0) -> GoalDirectedBenchmark:
    smiles = 'OC1(CN(C1)C(=O)C1=C(NC2=C(F)C=C(I)C=C2)C(F)=C(F)C=C1)C1CCCCN1'

    benchmark_name = f'hard_cobimetinib (max_logP: {max_logP})'
    modifier = ClippedScoreModifier(upper_x=0.7)
    os_tf = TanimotoScoringFunction(id=benchmark_name, target=smiles, fp_type='FCFP4', score_modifier=modifier)
    os_ap = TanimotoScoringFunction(id=benchmark_name, target=smiles, fp_type='ECFP6',
                                    score_modifier=MinGaussianModifier(mu=0.75, sigma=0.1))

    rot_b = RdkitScoringFunction(id=benchmark_name, descriptor=num_rotatable_bonds,
                                 score_modifier=MinGaussianModifier(mu=3, sigma=1))

    rings = RdkitScoringFunction(id=benchmark_name, descriptor=num_aromatic_rings,
                                 score_modifier=MaxGaussianModifier(mu=3, sigma=1))

    t_cns = ArithmeticMeanScoringFunction(id=benchmark_name,
                                          scoring_functions=[os_tf, os_ap, rot_b, rings,
                                                             CNS_MPO_ScoringFunction(id=benchmark_name, max_logP=max_logP)])

    specification = uniform_specification(1, 10, 100)

    return GoalDirectedBenchmark(name='Cobimetinib MPO',
                                 objective=t_cns,
                                 contribution_specification=specification)


def hard_osimertinib(mean_cls=GeometricMeanScoringFunction) -> GoalDirectedBenchmark:
    smiles = 'COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc2nccc(n2)c3cn(C)c4ccccc34'
    benchmark_name = f'hard_osimertinib (mean_cls: {mean_cls})'
    modifier = ClippedScoreModifier(upper_x=0.8)
    similar_to_osimertinib = TanimotoScoringFunction(id=benchmark_name, target=smiles,
                                                     fp_type='FCFP4', score_modifier=modifier)

    but_not_too_similar = TanimotoScoringFunction(id=benchmark_name, target=smiles, fp_type='ECFP6',
                                                  score_modifier=MinGaussianModifier(mu=0.85, sigma=0.1))

    tpsa_over_100 = RdkitScoringFunction(id=benchmark_name, descriptor=tpsa,
                                         score_modifier=MaxGaussianModifier(mu=100, sigma=10))

    logP_scoring = RdkitScoringFunction(id=benchmark_name, descriptor=logP,
                                        score_modifier=MinGaussianModifier(mu=1, sigma=1))

    make_osimertinib_great_again = mean_cls(id=benchmark_name, scoring_functions=[similar_to_osimertinib,
                                                                                  but_not_too_similar,
                                                                                  tpsa_over_100,
                                                                                  logP_scoring])

    specification = uniform_specification(1, 10, 100)

    return GoalDirectedBenchmark(name='Osimertinib MPO',
                                 objective=make_osimertinib_great_again,
                                 contribution_specification=specification)


def hard_fexofenadine(mean_cls=GeometricMeanScoringFunction) -> GoalDirectedBenchmark:
    """
    make fexofenadine less greasy
    :return:
    """
    smiles = 'CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4'
    benchmark_name = f'hard_fexofenadine (mean_cls: {mean_cls})'
    modifier = ClippedScoreModifier(upper_x=0.8)
    similar_to_fexofenadine = TanimotoScoringFunction(id=benchmark_name, target=smiles, fp_type='AP',
                                                      score_modifier=modifier)

    tpsa_over_90 = RdkitScoringFunction(id=benchmark_name, descriptor=tpsa,
                                        score_modifier=MaxGaussianModifier(mu=90, sigma=10))

    logP_under_4 = RdkitScoringFunction(id=benchmark_name, descriptor=logP,
                                        score_modifier=MinGaussianModifier(mu=4, sigma=1))

    optimize_fexofenadine = mean_cls(id=benchmark_name, scoring_functions=[similar_to_fexofenadine,
                                                                           tpsa_over_90,
                                                                           logP_under_4])

    specification = uniform_specification(1, 10, 100)

    return GoalDirectedBenchmark(name='Fexofenadine MPO',
                                 objective=optimize_fexofenadine,
                                 contribution_specification=specification)


def start_pop_ranolazine() -> GoalDirectedBenchmark:
    ranolazine = 'COc1ccccc1OCC(O)CN2CCN(CC(=O)Nc3c(C)cccc3C)CC2'
    benchmark_name = f'start_pop_ranolazine'
    modifier = ClippedScoreModifier(upper_x=0.7)
    similar_to_ranolazine = TanimotoScoringFunction(id=benchmark_name, target=ranolazine,
                                                    fp_type='AP', score_modifier=modifier)

    logP_under_4 = RdkitScoringFunction(id=benchmark_name, descriptor=logP,
                                        score_modifier=MaxGaussianModifier(mu=7, sigma=1))

    aroma = RdkitScoringFunction(id=benchmark_name, descriptor=num_aromatic_rings,
                                 score_modifier=MinGaussianModifier(mu=1, sigma=1))

    fluorine = RdkitScoringFunction(id=benchmark_name, descriptor=AtomCounter('F'),
                                    score_modifier=GaussianModifier(mu=1, sigma=1.0))

    optimize_ranolazine = ArithmeticMeanScoringFunction(id=benchmark_name, scoring_functions=[similar_to_ranolazine,
                                                                                              logP_under_4,
                                                                                              fluorine,
                                                                                              aroma])

    specification = uniform_specification(1, 10, 100)

    return GoalDirectedBenchmark(name='Ranolazine MPO',
                                 objective=optimize_ranolazine,
                                 contribution_specification=specification,
                                 starting_population=[ranolazine])


def weird_physchem() -> GoalDirectedBenchmark:
    benchmark_name = f'weird_physchem'
    min_bertz = RdkitScoringFunction(id=benchmark_name, descriptor=bertz,
                                     score_modifier=MaxGaussianModifier(mu=1500, sigma=200))

    mol_under_400 = RdkitScoringFunction(id=benchmark_name, descriptor=mol_weight,
                                         score_modifier=MinGaussianModifier(mu=400, sigma=40))

    aroma = RdkitScoringFunction(id=benchmark_name,descriptor=num_aromatic_rings,
                                 score_modifier=MinGaussianModifier(mu=3, sigma=1))

    fluorine = RdkitScoringFunction(id=benchmark_name, descriptor=AtomCounter('F'),
                                    score_modifier=GaussianModifier(mu=6, sigma=1.0))

    opt_weird = ArithmeticMeanScoringFunction(id=benchmark_name, scoring_functions=[min_bertz,
                                                                                    mol_under_400,
                                                                                    aroma, fluorine])

    specification = uniform_specification(1, 10, 100)

    return GoalDirectedBenchmark(name='Physchem MPO',
                                 objective=opt_weird,
                                 contribution_specification=specification)


def similarity_cns_mpo(smiles, molecule_name, max_logP=5.0) -> GoalDirectedBenchmark:
    benchmark_name = f'similarity_cns_mpo (smiles:{smiles} name:{molecule_name} max_logP:{max_logP})'
    os_tf = TanimotoScoringFunction(id=benchmark_name, target=smiles, fp_type='FCFP4')
    os_ap = TanimotoScoringFunction(id=benchmark_name, target=smiles, fp_type='AP')
    anti_fp = TanimotoScoringFunction(id=benchmark_name, target=smiles, fp_type='ECFP6',
                                      score_modifier=MinGaussianModifier(mu=0.70, sigma=0.1))

    t_cns = ArithmeticMeanScoringFunction(id=benchmark_name,
                                          scoring_functions=[os_tf, os_ap, anti_fp,
                                                             CNS_MPO_ScoringFunction(id=benchmark_name, max_logP=max_logP)])

    specification = uniform_specification(1, 10, 100)

    return GoalDirectedBenchmark(name=benchmark_name,
                                 objective=t_cns,
                                 contribution_specification=specification)


def similarity(smiles: str, name: str, fp_type: str = 'ECFP4', threshold: float = 0.7,
               rediscovery: bool = False) -> GoalDirectedBenchmark:

    category = 'rediscovery' if rediscovery else 'similarity'
    benchmark_name = f'similarity ( name:{name} category:{category} ' \
                     f'fp_type:{fp_type} threshold:{threshold})'

    modifier = ClippedScoreModifier(upper_x=threshold)
    scoring_function = TanimotoScoringFunction(id=benchmark_name, target=smiles, fp_type=fp_type,
                                               score_modifier=modifier)
    if rediscovery:
        specification = uniform_specification(1)
    else:
        specification = uniform_specification(1, 10, 100)

    return GoalDirectedBenchmark(name=benchmark_name,
                                 objective=scoring_function,
                                 contribution_specification=specification)


def logP_benchmark(target: float) -> GoalDirectedBenchmark:
    benchmark_name = f'logP (target: {target})'
    objective = RdkitScoringFunction(id=benchmark_name,
                                     descriptor=logP,
                                     score_modifier=GaussianModifier(mu=target, sigma=1))

    specification = uniform_specification(1, 10, 100)

    return GoalDirectedBenchmark(name=benchmark_name,
                                 objective=objective,
                                 contribution_specification=specification)


def tpsa_benchmark(target: float) -> GoalDirectedBenchmark:
    benchmark_name = f'TPSA (target: {target})'
    objective = RdkitScoringFunction(id=benchmark_name,
                                     descriptor=tpsa,
                                     score_modifier=GaussianModifier(mu=target, sigma=20.0))

    specification = uniform_specification(1, 10, 100)

    return GoalDirectedBenchmark(name=benchmark_name,
                                 objective=objective,
                                 contribution_specification=specification)


def cns_mpo(max_logP=5.0) -> GoalDirectedBenchmark:
    benchmark_name = f'cns (max_logP: {max_logP})'
    objective = CNS_MPO_ScoringFunction(id=benchmark_name, max_logP=max_logP)

    specification = uniform_specification(1, 10, 100)

    return GoalDirectedBenchmark(name='CNS MPO', objective=objective,
                                 contribution_specification=specification)


def qed_benchmark() -> GoalDirectedBenchmark:
    benchmark_name = f'qed'
    specification = uniform_specification(1, 10, 100)

    objective = RdkitScoringFunction(id=benchmark_name, descriptor=qed)

    return GoalDirectedBenchmark(name='QED',
                                 objective=objective,
                                 contribution_specification=specification)


def median_camphor_menthol(mean_cls=GeometricMeanScoringFunction) -> GoalDirectedBenchmark:
    benchmark_name = f'median_camphor_menthol (mean_cls: {mean_cls})'
    t_camphor = TanimotoScoringFunction(id=benchmark_name, target='CC1(C)C2CCC1(C)C(=O)C2', fp_type='ECFP4')
    t_menthol = TanimotoScoringFunction(id=benchmark_name, target='CC(C)C1CCC(C)CC1O', fp_type='ECFP4')
    median = mean_cls(id=benchmark_name, scoring_functions=[t_menthol, t_camphor])

    specification = uniform_specification(1, 10, 100)

    return GoalDirectedBenchmark(name='Median molecules 1',
                                 objective=median,
                                 contribution_specification=specification)


# def novelty_benchmark(training_set_file: str, number_samples: int) -> DistributionLearningBenchmark:
#    smiles_list = [s.strip() for s in open(training_set_file).readlines()]
#    return NoveltyBenchmark(number_samples=number_samples, training_set=smiles_list)


# def kldiv_benchmark(training_set_file: str, number_samples: int) -> DistributionLearningBenchmark:
#    smiles_list = [s.strip() for s in open(training_set_file).readlines()]
#    return KLDivBenchmark(number_samples=number_samples, training_set=smiles_list)


# def frechet_benchmark(training_set_file: str, number_samples: int) -> DistributionLearningBenchmark:
#    smiles_list = [s.strip() for s in open(training_set_file).readlines()]
#    return FrechetBenchmark(training_set=smiles_list, sample_size=number_samples)


def perindopril_rings() -> GoalDirectedBenchmark:
    # perindopril with two aromatic rings
    benchmark_name = f'perindopril_rings'
    perindopril = TanimotoScoringFunction(id=benchmark_name, target='O=C(OCC)C(NC(C(=O)N1C(C(=O)O)CC2CCCCC12)C)CCC',
                                          fp_type='ECFP4')

    arom_rings = RdkitScoringFunction(id=benchmark_name, descriptor=num_aromatic_rings,
                                      score_modifier=GaussianModifier(mu=2, sigma=0.5))
    objective = GeometricMeanScoringFunction(id=benchmark_name, scoring_functions=[perindopril, arom_rings])
    specification = uniform_specification(1, 10, 100)

    return GoalDirectedBenchmark(name='Perindopril MPO',
                                 objective=objective,
                                 contribution_specification=specification)


def amlodipine_rings() -> GoalDirectedBenchmark:
    # amlodipine with 3 rings
    benchmark_name = f'amlodipine_rings' 
    amlodipine = TanimotoScoringFunction(id=benchmark_name,
                                         target=r'Clc1ccccc1C2C(=C(/N/C(=C2/C(=O)OCC)COCCN)C)\C(=O)OC',
                                         fp_type='ECFP4')
    rings = RdkitScoringFunction(id=benchmark_name, descriptor=num_rings,
                                 score_modifier=GaussianModifier(mu=3, sigma=0.5))
    objective = GeometricMeanScoringFunction(id=benchmark_name, scoring_functions=[amlodipine, rings])
    specification = uniform_specification(1, 10, 100)

    return GoalDirectedBenchmark(name='Amlodipine MPO',
                                 objective=objective,
                                 contribution_specification=specification)


def sitagliptin_replacement() -> GoalDirectedBenchmark:
    # Find a molecule dissimilar to sitagliptin, but with the same properties
    benchmark_name = f'sitagliptin_replacement'
    smiles = 'Fc1cc(c(F)cc1F)CC(N)CC(=O)N3Cc2nnc(n2CC3)C(F)(F)F'
    sitagliptin = Chem.MolFromSmiles(smiles)
    target_logp = logP(sitagliptin)
    target_tpsa = tpsa(sitagliptin)

    similarity = TanimotoScoringFunction(id=benchmark_name, target=smiles, fp_type='ECFP4',
                                         score_modifier=GaussianModifier(mu=0, sigma=0.1))
    lp = RdkitScoringFunction(id=benchmark_name, descriptor=logP,
                              score_modifier=GaussianModifier(mu=target_logp, sigma=0.2))
    tp = RdkitScoringFunction(id=benchmark_name, descriptor=tpsa,
                              score_modifier=GaussianModifier(mu=target_tpsa, sigma=5))
    isomers = IsomerScoringFunction(id=benchmark_name, molecular_formula='C16H15F6N5O')
    objective = GeometricMeanScoringFunction(id=benchmark_name, scoring_functions=[similarity, lp, tp, isomers])
    specification = uniform_specification(1, 10, 100)

    return GoalDirectedBenchmark(name='Sitagliptin MPO',
                                 objective=objective,
                                 contribution_specification=specification)


def zaleplon_with_other_formula() -> GoalDirectedBenchmark:
    # zaleplon_with_other_formula with other formula
    benchmark_name = f'zaleplon_with_other_formula'
    zaleplon = TanimotoScoringFunction(id=benchmark_name, target='O=C(C)N(CC)C1=CC=CC(C2=CC=NC3=C(C=NN23)C#N)=C1',
                                       fp_type='ECFP4')
    formula = IsomerScoringFunction(id=benchmark_name, molecular_formula='C19H17N3O2')
    objective = GeometricMeanScoringFunction(id=benchmark_name, scoring_functions=[zaleplon, formula])
    specification = uniform_specification(1, 10, 100)

    return GoalDirectedBenchmark(name='Zaleplon MPO',
                                 objective=objective,
                                 contribution_specification=specification)


def smarts_with_other_target(smarts: str, other_molecule: str) -> ScoringFunction:
    benchmark_name = f'smarts_with_other_target (smarts:{smarts} other_molecule:{other_molecule})'
    smarts_scoring_function = SMARTSScoringFunction(id=benchmark_name, target=smarts)
    other_mol = Chem.MolFromSmiles(other_molecule)
    target_logp = logP(other_mol)
    target_tpsa = tpsa(other_mol)
    target_bertz = bertz(other_mol)

    lp = RdkitScoringFunction(id=benchmark_name, descriptor=logP,
                              score_modifier=GaussianModifier(mu=target_logp, sigma=0.2))
    tp = RdkitScoringFunction(id=benchmark_name,descriptor=tpsa,
                              score_modifier=GaussianModifier(mu=target_tpsa, sigma=5))
    bz = RdkitScoringFunction(id=benchmark_name,descriptor=bertz,
                              score_modifier=GaussianModifier(mu=target_bertz, sigma=30))

    return GeometricMeanScoringFunction(id=benchmark_name, scoring_functions=[smarts_scoring_function, lp, tp, bz])


def valsartan_smarts() -> GoalDirectedBenchmark:
    # valsartan smarts with sitagliptin properties
    benchmark_name = f'valsartan_smarts'
    sitagliptin_smiles = 'NC(CC(=O)N1CCn2c(nnc2C(F)(F)F)C1)Cc1cc(F)c(F)cc1F'
    valsartan_smarts = 'CN(C=O)Cc1ccc(c2ccccc2)cc1'
    objective = smarts_with_other_target(valsartan_smarts, sitagliptin_smiles)
    objective.id = benchmark_name
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(name='Valsartan SMARTS',
                                 objective=objective,
                                 contribution_specification=specification)


def median_tadalafil_sildenafil() -> GoalDirectedBenchmark:
    # median mol between tadalafil and sildenafil
    benchmark_name = f'median_tadalafil_sildenafil'
    m1 = TanimotoScoringFunction(id=benchmark_name,
                                 target='O=C1N(CC(N2C1CC3=C(C2C4=CC5=C(OCO5)C=C4)NC6=C3C=CC=C6)=O)C', fp_type='ECFP6')
    m2 = TanimotoScoringFunction(id=benchmark_name,
                                 target='CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C', fp_type='ECFP6')
    median = GeometricMeanScoringFunction(id=benchmark_name, scoring_functions=[m1, m2])

    specification = uniform_specification(1, 10, 100)

    return GoalDirectedBenchmark(name='Median molecules 2',
                                 objective=median,
                                 contribution_specification=specification)


def pioglitazone_mpo() -> GoalDirectedBenchmark:
    # pioglitazone with same mw but less rotatable bonds
    benchmark_name = f'pioglitazone'
    smiles = 'O=C1NC(=O)SC1Cc3ccc(OCCc2ncc(cc2)CC)cc3'
    pioglitazone = Chem.MolFromSmiles(smiles)
    target_molw = mol_weight(pioglitazone)

    similarity = TanimotoScoringFunction(id=benchmark_name, target=smiles, fp_type='ECFP4',
                                         score_modifier=GaussianModifier(mu=0, sigma=0.1))
    mw = RdkitScoringFunction(id=benchmark_name, descriptor=mol_weight, score_modifier=GaussianModifier(mu=target_molw, sigma=10))
    rb = RdkitScoringFunction(id=benchmark_name, descriptor=num_rotatable_bonds, score_modifier=GaussianModifier(mu=2, sigma=0.5))
    objective = GeometricMeanScoringFunction(id=benchmark_name, scoring_functions=[similarity, mw, rb])
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(name=benchmark_name,
                                 objective=objective,
                                 contribution_specification=specification)


def decoration_hop() -> GoalDirectedBenchmark:
    smiles = 'CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)(C)C'
    benchmark_name = f'decoration_hop'
    pharmacophor_sim = TanimotoScoringFunction(id=benchmark_name, target=smiles, fp_type='PHCO',
                                               score_modifier=ClippedScoreModifier(upper_x=0.85))
    # change deco
    deco1 = SMARTSScoringFunction(id=benchmark_name, target='CS([#6])(=O)=O', inverse=True)
    deco2 = SMARTSScoringFunction(id=benchmark_name, target='[#7]-c1ccc2ncsc2c1', inverse=True)

    # keep scaffold
    scaffold = SMARTSScoringFunction(id=benchmark_name, target='[#7]-c1n[c;h1]nc2[c;h1]c(-[#8])[c;h0][c;h1]c12',
                                     inverse=False)

    deco_hop1_fn = ArithmeticMeanScoringFunction(id=benchmark_name, scoring_functions=[pharmacophor_sim, deco1, deco2,
                                                                                       scaffold])

    specification = uniform_specification(1, 10, 100)

    return GoalDirectedBenchmark(name='Deco Hop',
                                 objective=deco_hop1_fn,
                                 contribution_specification=specification)


def scaffold_hop() -> GoalDirectedBenchmark:
    """
    Keep the decoration, and similarity to start point, but change the scaffold.
    """
    benchmark_name = f'scaffold_hop'
    smiles = 'CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)(C)C'

    pharmacophor_sim = TanimotoScoringFunction(id=benchmark_name, target=smiles, fp_type='PHCO',
                                               score_modifier=ClippedScoreModifier(upper_x=0.75))

    deco = SMARTSScoringFunction(id=benchmark_name,
                                 target='[#6]-[#6]-[#6]-[#8]-[#6]~[#6]~[#6]~[#6]~[#6]-[#7]-c1ccc2ncsc2c1',
                                 inverse=False)

    # anti scaffold
    scaffold = SMARTSScoringFunction(id=benchmark_name,
                                     target='[#7]-c1n[c;h1]nc2[c;h1]c(-[#8])[c;h0][c;h1]c12',
                                     inverse=True)

    scaffold_hop_obj = ArithmeticMeanScoringFunction(id=benchmark_name, scoring_functions=[pharmacophor_sim,
                                                                                           deco, scaffold])

    specification = uniform_specification(1, 10, 100)

    return GoalDirectedBenchmark(name='Scaffold Hop',
                                 objective=scaffold_hop_obj,
                                 contribution_specification=specification)


def ranolazine_mpo() -> GoalDirectedBenchmark:
    """
    Make start_pop_ranolazine more polar and add a fluorine
    """
    ranolazine = 'COc1ccccc1OCC(O)CN2CCN(CC(=O)Nc3c(C)cccc3C)CC2'
    benchmark_name = f'ranolazine_mpo'

    modifier = ClippedScoreModifier(upper_x=0.7)
    similar_to_ranolazine = TanimotoScoringFunction(id=benchmark_name, target=ranolazine, fp_type='AP',
                                                    score_modifier=modifier)

    logP_under_4 = RdkitScoringFunction(id=benchmark_name, descriptor=logP,
                                        score_modifier=MaxGaussianModifier(mu=7, sigma=1))

    tpsa_f = RdkitScoringFunction(id=benchmark_name, descriptor=tpsa,
                                  score_modifier=MaxGaussianModifier(mu=95, sigma=20))

    fluorine = RdkitScoringFunction(id=benchmark_name, descriptor=AtomCounter('F'),
                                    score_modifier=GaussianModifier(mu=1, sigma=1.0))

    optimize_ranolazine = GeometricMeanScoringFunction(id=benchmark_name,
                                                       scoring_functions=[similar_to_ranolazine, logP_under_4,
                                                                          fluorine, tpsa_f])

    specification = uniform_specification(1, 10, 100)

    return GoalDirectedBenchmark(name='Ranolazine MPO',
                                 objective=optimize_ranolazine,
                                 contribution_specification=specification,
                                 starting_population=[ranolazine])
