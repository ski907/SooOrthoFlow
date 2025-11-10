#!/usr/bin/env python3
"""
Utility to manage camera calibrations across dated calibration files
"""
import pickle
import argparse
from pathlib import Path
import shutil


def list_cameras(cal_file):
    """List all cameras in a calibration file"""
    with open(cal_file, 'rb') as f:
        calibrations = pickle.load(f)

    print(f"\nCameras in {cal_file.name}:")
    print("=" * 60)
    for i, (camera_id, cal_data) in enumerate(calibrations.items(), 1):
        recal_date = cal_data.get('recalibration_date', 'original')
        rms = cal_data.get('rms', 'N/A')
        n_gcps = cal_data.get('n_gcps', 'N/A')
        print(f"{i:2}. {camera_id:30} | Date: {recal_date:8} | RMS: {rms:6} | GCPs: {n_gcps}")
    print("=" * 60)
    print(f"Total: {len(calibrations)} cameras\n")


def copy_camera(source_file, dest_file, camera_id):
    """
    Copy a camera's calibration from source file to destination file

    Args:
        source_file: Path to source calibration file
        dest_file: Path to destination calibration file
        camera_id: Camera to copy
    """
    # Load both files
    with open(source_file, 'rb') as f:
        source_cals = pickle.load(f)

    with open(dest_file, 'rb') as f:
        dest_cals = pickle.load(f)

    # Check if camera exists in source
    if camera_id not in source_cals:
        print(f"✗ Error: Camera '{camera_id}' not found in {source_file.name}")
        print(f"Available cameras: {', '.join(source_cals.keys())}")
        return False

    # Create backup of destination
    backup_file = dest_file.parent / f"{dest_file.stem}_backup_before_restore.pkl"
    shutil.copy2(dest_file, backup_file)
    print(f"Created backup: {backup_file.name}")

    # Copy camera calibration
    dest_cals[camera_id] = source_cals[camera_id]

    # Save updated destination
    with open(dest_file, 'wb') as f:
        pickle.dump(dest_cals, f)

    print(f"✓ Copied {camera_id} from {source_file.name} to {dest_file.name}")
    return True


def delete_camera(cal_file, camera_id):
    """
    Delete a camera's calibration from a file

    Args:
        cal_file: Path to calibration file
        camera_id: Camera to delete
    """
    with open(cal_file, 'rb') as f:
        calibrations = pickle.load(f)

    if camera_id not in calibrations:
        print(f"✗ Error: Camera '{camera_id}' not found in {cal_file.name}")
        print(f"Available cameras: {', '.join(calibrations.keys())}")
        return False

    # Create backup
    backup_file = cal_file.parent / f"{cal_file.stem}_backup_before_delete.pkl"
    shutil.copy2(cal_file, backup_file)
    print(f"Created backup: {backup_file.name}")

    # Delete camera
    del calibrations[camera_id]

    # Save
    with open(cal_file, 'wb') as f:
        pickle.dump(calibrations, f)

    print(f"✓ Deleted {camera_id} from {cal_file.name}")
    return True


def restore_from_backup(backup_file, camera_id, dest_file):
    """
    Restore a camera from a backup file

    Args:
        backup_file: Path to backup file (e.g., *_backup_camera_name_20251016.pkl)
        camera_id: Camera to restore
        dest_file: Destination calibration file
    """
    # The backup file contains just the one camera
    with open(backup_file, 'rb') as f:
        backup_cals = pickle.load(f)

    if camera_id not in backup_cals:
        print(f"✗ Error: Camera '{camera_id}' not found in backup {backup_file.name}")
        print(f"Available cameras in backup: {', '.join(backup_cals.keys())}")
        return False

    # Load destination
    with open(dest_file, 'rb') as f:
        dest_cals = pickle.load(f)

    # Create backup of destination before restoring
    safety_backup = dest_file.parent / f"{dest_file.stem}_backup_before_restore.pkl"
    shutil.copy2(dest_file, safety_backup)
    print(f"Created safety backup: {safety_backup.name}")

    # Restore camera
    dest_cals[camera_id] = backup_cals[camera_id]

    # Save
    with open(dest_file, 'wb') as f:
        pickle.dump(dest_cals, f)

    print(f"✓ Restored {camera_id} from {backup_file.name} to {dest_file.name}")
    return True


def interactive_mode(cal_dir):
    """Interactive mode to manage calibrations"""
    cal_dir = Path(cal_dir)

    # Find all calibration files
    cal_files = sorted(cal_dir.glob('camera_calibrations*.pkl'))

    if not cal_files:
        print(f"No calibration files found in {cal_dir}")
        return

    print("\nCalibration File Manager")
    print("=" * 60)

    while True:
        print("\nOptions:")
        print("1. List cameras in a file")
        print("2. Copy camera from one file to another")
        print("3. Delete camera from a file")
        print("4. Restore camera from backup")
        print("5. Quit")

        choice = input("\nChoice (1-5): ").strip()

        if choice == '1':
            # List files
            print("\nAvailable calibration files:")
            for i, cal_file in enumerate(cal_files, 1):
                print(f"  {i}. {cal_file.name}")

            file_choice = input(f"\nSelect file (1-{len(cal_files)}): ").strip()
            try:
                cal_file = cal_files[int(file_choice) - 1]
                list_cameras(cal_file)
            except (ValueError, IndexError):
                print("Invalid choice")

        elif choice == '2':
            # Copy camera
            print("\nAvailable calibration files:")
            for i, cal_file in enumerate(cal_files, 1):
                print(f"  {i}. {cal_file.name}")

            src_choice = input(f"\nSelect SOURCE file (1-{len(cal_files)}): ").strip()
            dst_choice = input(f"Select DESTINATION file (1-{len(cal_files)}): ").strip()

            try:
                source_file = cal_files[int(src_choice) - 1]
                dest_file = cal_files[int(dst_choice) - 1]

                # Show cameras in source
                with open(source_file, 'rb') as f:
                    source_cals = pickle.load(f)

                print(f"\nCameras in {source_file.name}:")
                for i, cam in enumerate(source_cals.keys(), 1):
                    print(f"  {i}. {cam}")

                cam_choice = input(f"\nSelect camera to copy (1-{len(source_cals)}): ").strip()
                camera_id = list(source_cals.keys())[int(cam_choice) - 1]

                copy_camera(source_file, dest_file, camera_id)

            except (ValueError, IndexError):
                print("Invalid choice")

        elif choice == '3':
            # Delete camera
            print("\nAvailable calibration files:")
            for i, cal_file in enumerate(cal_files, 1):
                print(f"  {i}. {cal_file.name}")

            file_choice = input(f"\nSelect file (1-{len(cal_files)}): ").strip()

            try:
                cal_file = cal_files[int(file_choice) - 1]

                # Show cameras
                with open(cal_file, 'rb') as f:
                    calibrations = pickle.load(f)

                print(f"\nCameras in {cal_file.name}:")
                for i, cam in enumerate(calibrations.keys(), 1):
                    print(f"  {i}. {cam}")

                cam_choice = input(f"\nSelect camera to delete (1-{len(calibrations)}): ").strip()
                camera_id = list(calibrations.keys())[int(cam_choice) - 1]

                confirm = input(f"Are you sure you want to delete {camera_id}? (yes/no): ").strip().lower()
                if confirm == 'yes':
                    delete_camera(cal_file, camera_id)
                else:
                    print("Cancelled")

            except (ValueError, IndexError):
                print("Invalid choice")

        elif choice == '4':
            # Restore from backup
            backup_files = sorted(cal_dir.glob('*_backup_*.pkl'))

            if not backup_files:
                print("No backup files found")
                continue

            print("\nAvailable backup files:")
            for i, backup_file in enumerate(backup_files, 1):
                print(f"  {i}. {backup_file.name}")

            backup_choice = input(f"\nSelect backup file (1-{len(backup_files)}): ").strip()

            try:
                backup_file = backup_files[int(backup_choice) - 1]

                # Show cameras in backup
                with open(backup_file, 'rb') as f:
                    backup_cals = pickle.load(f)

                print(f"\nCameras in backup:")
                for i, cam in enumerate(backup_cals.keys(), 1):
                    print(f"  {i}. {cam}")

                cam_choice = input(f"\nSelect camera to restore (1-{len(backup_cals)}): ").strip()
                camera_id = list(backup_cals.keys())[int(cam_choice) - 1]

                # Select destination
                print("\nAvailable calibration files:")
                for i, cal_file in enumerate(cal_files, 1):
                    print(f"  {i}. {cal_file.name}")

                dest_choice = input(f"\nSelect DESTINATION file (1-{len(cal_files)}): ").strip()
                dest_file = cal_files[int(dest_choice) - 1]

                restore_from_backup(backup_file, camera_id, dest_file)

            except (ValueError, IndexError):
                print("Invalid choice")

        elif choice == '5':
            print("Exiting")
            break

        else:
            print("Invalid choice")


def main():
    parser = argparse.ArgumentParser(
        description='Manage camera calibrations across dated files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python manage_calibrations.py

  # List cameras in a file
  python manage_calibrations.py --list camera_calibrations_20251016.pkl

  # Copy camera from old file to new file
  python manage_calibrations.py --copy camera_calibrations.pkl camera_calibrations_20251016.pkl NVR2_N910A6_ch1_main

  # Restore camera from backup
  python manage_calibrations.py --restore camera_calibrations_backup_NVR2_N910A6_ch1_main_20251016.pkl NVR2_N910A6_ch1_main camera_calibrations_20251016.pkl
        """
    )

    parser.add_argument('--list', '-l', metavar='FILE',
                       help='List all cameras in a calibration file')
    parser.add_argument('--copy', '-c', nargs=3, metavar=('SOURCE', 'DEST', 'CAMERA'),
                       help='Copy camera calibration from source to destination')
    parser.add_argument('--delete', '-d', nargs=2, metavar=('FILE', 'CAMERA'),
                       help='Delete camera from calibration file')
    parser.add_argument('--restore', '-r', nargs=3, metavar=('BACKUP', 'CAMERA', 'DEST'),
                       help='Restore camera from backup to destination file')
    parser.add_argument('--dir', default='calibration',
                       help='Calibration directory (default: calibration)')

    args = parser.parse_args()

    cal_dir = Path(args.dir)

    if args.list:
        list_cameras(Path(args.list))
    elif args.copy:
        source, dest, camera = args.copy
        copy_camera(Path(source), Path(dest), camera)
    elif args.delete:
        file, camera = args.delete
        delete_camera(Path(file), camera)
    elif args.restore:
        backup, camera, dest = args.restore
        restore_from_backup(Path(backup), camera, Path(dest))
    else:
        # Interactive mode
        interactive_mode(cal_dir)


if __name__ == '__main__':
    main()
