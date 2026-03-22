import tempfile

import lightning as L
import pytest

from ticl.fit_model import main
from ticl.models.tabflex import TabFlex
from ticl.prediction import TabPFNClassifier

from ticl.testing_utils import count_parameters, check_predict_iris

TESTING_DEFAULTS = ['-C', '-E', '10', '-n', '1', '-A', 'False', '-e', '128', '-N', '4', '--experiment',
                    'testing_experiment', '--train-mixed-precision', 'False', '--validate', 'False']
TESTING_DEFAULTS_SHORT = ['-C', '-E', '2', '-n', '1', '-A', 'False', '-e', '128', '-N', '4', '--experiment',
                          'testing_experiment', '--train-mixed-precision', 'False',
                          '--save-every', '2', '--validate', 'False']


def test_train_tabflex_basic():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(['tabflex'] + TESTING_DEFAULTS + ['-B', tmpdir])
        clf = TabPFNClassifier(device='cpu', model_string=results['model_string'], epoch=results['epoch'], base_path=results['base_path'])
        check_predict_iris(clf)
    assert isinstance(results['model'], TabFlex)
    assert count_parameters(results['model']) == 580106
    assert results['model_string'].startswith("tabflex_AFalse_e128_E10_N4_n1_tFalse_cpu")
    assert results['loss'] == pytest.approx(0.6813156008720398, rel=1e-4)


def test_train_tabflex_identity():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(['tabflex'] + TESTING_DEFAULTS + ['-B', tmpdir, '--feature-map', 'identity_for_real'])
        clf = TabPFNClassifier(device='cpu', model_string=results['model_string'], epoch=results['epoch'], base_path=results['base_path'])
        check_predict_iris(clf)
    assert isinstance(results['model'], TabFlex)
    assert count_parameters(results['model']) == 580106
    assert results['model_string'].startswith("tabflex_AFalse_e128_E10_featuremapidentity_for_real_N4_n1_tFalse_cpu")
    assert results['loss'] == pytest.approx(0.6912140846252441, rel=1e-4)


def test_train_tabflex_hedgehog():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(['tabflex'] + TESTING_DEFAULTS + ['-B', tmpdir, '--feature-map', 'hedgehog'])
        clf = TabPFNClassifier(device='cpu', model_string=results['model_string'], epoch=results['epoch'], base_path=results['base_path'])
        check_predict_iris(clf)
    assert isinstance(results['model'], TabFlex)
    assert count_parameters(results['model']) == 646154
    assert results['model_string'].startswith("tabflex_AFalse_e128_E10_featuremaphedgehog_N4_n1_tFalse_cpu")
    assert results['loss'] == pytest.approx(0.7568668127059937, rel=1e-4)



def test_train_tabflex_hedgehog_shared():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(['tabflex'] + TESTING_DEFAULTS + ['-B', tmpdir, '--feature-map', 'hedgehog_shared'])
        clf = TabPFNClassifier(device='cpu', model_string=results['model_string'], epoch=results['epoch'], base_path=results['base_path'])
        check_predict_iris(clf)
    assert isinstance(results['model'], TabFlex)
    assert count_parameters(results['model']) == 596618
    assert results['model_string'].startswith("tabflex_AFalse_e128_E10_featuremaphedgehog_shared_N4_n1_tFalse_cpu")
    assert results['loss'] == pytest.approx(1.3891624212265015, rel=1e-4)



def test_train_tabflex_num_features():
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(['tabflex'] + TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '--num-features', '13'])
        clf = TabPFNClassifier(device='cpu', model_string=results['model_string'], epoch=results['epoch'], base_path=results['base_path'])
        check_predict_iris(clf)
    assert isinstance(results['model'], TabFlex)
    assert results['model'].encoder.weight.shape[1] == 13
    assert count_parameters(results['model']) == 568970
    assert results['loss'] == pytest.approx(0.7209413647651672, rel=1e-5)


def test_train_tabflex_num_samples():
    # smoke test only since I'm too lazy to mock
    L.seed_everything(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results = main(['tabflex'] + TESTING_DEFAULTS_SHORT + ['-B', tmpdir, '--n-samples', '35'])
        clf = TabPFNClassifier(device='cpu', model_string=results['model_string'], epoch=results['epoch'], base_path=results['base_path'])
        check_predict_iris(clf)
    assert isinstance(results['model'], TabFlex)
    assert count_parameters(results['model']) == 580106
    assert results['loss'] == pytest.approx(0.7025600075721741, rel=1e-5)

