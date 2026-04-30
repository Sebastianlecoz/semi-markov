"""
Model loading utilities for football sequence simulations.

This module provides functions to load Accelerated Failure Time (AFT) models
and the next-features model used in the simulation pipeline. The loaders
attempt multiple deserialization strategies to maximise robustness when
reading models from disk and provide informative console output.
"""


import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib

from . import console  # Shared rich console for styled output



def load_aft_models() -> Dict[str, Any]:
    """
    Load all AFT models from the ``models/aft`` directory.

    AFT models may be serialized via either ``joblib`` or the standard
    ``pickle`` module. This loader attempts each method in turn and falls
    back to latin-1 encoded pickle loading when necessary. Spline models
    and scalers are explicitly ignored based on filename patterns.

    Returns
    -------
    Dict[str, Any]
        A dictionary mapping the stem of each model file to its loaded
        object. If no models are found, an empty dictionary is returned.
    """
    aft_models: Dict[str, Any] = {}
    
    # Chercher le répertoire des modèles de façon robuste
    possible_paths = [
        Path('../../../models/aft'),  # Chemin relatif original
        Path('models/aft'),           # Depuis la racine du projet
        Path('/home/firasboustila/smc/models/aft'),  # Chemin absolu
    ]
    
    model_directory = None
    for path in possible_paths:
        if path.exists():
            model_directory = path
            break
    
    debug_log: List[str] = []
    if model_directory is None:
        console.print(f"[yellow]AFT model directory not found in any of: {possible_paths}[/yellow]")
        with open('debug_aft.log', 'w') as log_file:
            log_file.write(f"Directory not found in any of: {possible_paths}\n")
        return aft_models
    debug_log.append(f"Directory found: {model_directory}")
    # Gather candidate model files, skipping splines and the scaler
    candidate_files = [file for file in model_directory.glob('*.pkl')
                       if not file.name.endswith('_spline.pkl') and file.name != 'dist_ang_scaler.pkl']
    debug_log.append(f"Found {len(candidate_files)} AFT pickle files (excluding splines)")
    for model_file in candidate_files:
        debug_log.append(f"\nAttempting to load: {model_file.name}")
        model_obj: Optional[Any] = None
        # Try joblib first
        try:
            model_obj = joblib.load(model_file)
            debug_log.append(f"  ✓ Loaded via joblib: {type(model_obj)}")
        except Exception as exc_joblib:
            debug_log.append(f"  ✗ Joblib load failed: {exc_joblib}")
            # Fallback to standard pickle
            try:
                with open(model_file, 'rb') as fh:
                    model_obj = pickle.load(fh)
                debug_log.append(f"  ✓ Loaded via pickle: {type(model_obj)}")
            except Exception as exc_pickle:
                debug_log.append(f"  ✗ Standard pickle failed: {exc_pickle}")
                # Final fallback: latin-1 encoding
                try:
                    with open(model_file, 'rb') as fh:
                        model_obj = pickle.load(fh, encoding='latin-1')
                    debug_log.append(f"  ✓ Loaded via latin-1: {type(model_obj)}")
                except Exception as exc_latin:
                    debug_log.append(f"  ✗ Latin-1 load failed: {exc_latin}")
                    continue  # Skip this model
        if model_obj is not None:
            model_name = model_file.stem
            aft_models[model_name] = model_obj
            debug_log.append(f"  ✓ Added AFT model: {model_name}")
            console.print(f"[green]✓ AFT model loaded: {model_name}[/green]")
    debug_log.append(f"\nSUMMARY: {len(aft_models)} AFT models loaded")
    debug_log.append(f"Model keys: {list(aft_models.keys())}")
    # Persist the debug log
    with open('debug_aft.log', 'w') as log_file:
        log_file.write('\n'.join(debug_log))
    console.print(f"[green]Total AFT models loaded: {len(aft_models)}[/green]")
    return aft_models
