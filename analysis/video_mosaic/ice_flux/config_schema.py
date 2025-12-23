"""
Configuration validation for ice flux analysis.

Author: SooOrthoFlow Team
"""

from typing import Dict, List, Tuple, Any


def validate_ice_flux_config(config: Dict) -> Tuple[bool, List[str]]:
    """
    Validate ice flux configuration.

    Parameters:
        config: Ice flux configuration dictionary

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    if not config.get('enabled', False):
        return True, []

    # Check farneback_params
    if 'farneback_params' not in config:
        errors.append("farneback_params is required when enabled=true")
    else:
        params = config['farneback_params']
        required_params = ['pyr_scale', 'levels', 'winsize', 'iterations', 'poly_n', 'poly_sigma', 'flags']
        for param in required_params:
            if param not in params:
                errors.append(f"farneback_params.{param} is required")

        # Validate parameter ranges
        if 'pyr_scale' in params and not (0 < params['pyr_scale'] < 1):
            errors.append("farneback_params.pyr_scale must be between 0 and 1 (exclusive)")

        if 'levels' in params and params['levels'] < 1:
            errors.append("farneback_params.levels must be >= 1")

        if 'winsize' in params and params['winsize'] < 3:
            errors.append("farneback_params.winsize must be >= 3")

        if 'iterations' in params and params['iterations'] < 1:
            errors.append("farneback_params.iterations must be >= 1")

        if 'poly_n' in params and params['poly_n'] not in [5, 7]:
            errors.append("farneback_params.poly_n must be 5 or 7")

    # Check validation plot interval
    if 'validation_plot_interval' in config and config['validation_plot_interval'] < 1:
        errors.append("validation_plot_interval must be >= 1")

    # Check overlay video subsample
    if 'overlay_video_subsample' in config and config['overlay_video_subsample'] < 1:
        errors.append("overlay_video_subsample must be >= 1")

    return (len(errors) == 0, errors)


def get_default_config() -> Dict[str, Any]:
    """
    Get default ice flux configuration.

    Returns:
        Default configuration dictionary
    """
    return {
        'enabled': False,
        'farneback_params': {
            'pyr_scale': 0.5,
            'levels': 3,
            'winsize': 15,
            'iterations': 3,
            'poly_n': 5,
            'poly_sigma': 1.2,
            'flags': 0
        },
        'save_velocity_geotiffs': True,
        'create_validation_plots': True,
        'validation_plot_interval': 10,
        'create_overlay_video': False,
        'overlay_video_subsample': 20,
        'max_arrow_velocity': 0.5,
        'compress_geotiffs': True,
        'optional': True
    }
