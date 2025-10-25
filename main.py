#!/usr/bin/env python3
"""
Main script launcher for UCC update workflow.

This script provides an interactive menu to run the sequential UCC update scripts:
A -> B -> C -> D

Each script must be run in order for proper UCC database maintenance.
"""

import sys

# Import modules
from modules import (
    A_get_new_DB,
    B_update_UCC_DB,
    C_process_member_files,
    D_update_UCC_site,
)


def display_menu():
    """Display the main menu options."""
    print("\n" + "=" * 60)
    print("UCC Update Workflow - Script Launcher")
    print("=" * 60)
    print("\nAvailable scripts (must be run in sequence A → B → C → D):")
    print("\nA) A_get_new_DB.py")
    print("   └─ Add a new database to the UCC")
    print("   └─ Fetches and validates new astronomical databases")
    print("   └─ ⚠️  Requires NASA/ADS bibcode input")

    print("\nB) B_update_UCC_DB.py")
    print("   └─ Update the UCC database")
    print("   └─ Integrates new data with existing UCC catalog")

    print("\nC) C_process_member_files.py")
    print("   └─ Process member files and update catalogue")
    print("   └─ Processes stellar membership using fastMP algorithm")

    print("\nD) D_update_UCC_site.py")
    print("   └─ Update the website files")
    print("   └─ Generates website content and visualization files")

    print("\nOther options:")
    print("Q) Quit")
    print("\n" + "=" * 60)


def get_ads_bibcode():
    """Get ADS bibcode from user input with validation."""
    print("\n📚 NASA/ADS Bibcode Required")
    print("Examples:")
    print("  • 2018MNRAS.481.3902B")
    print("  • 2021A&A...652A.102C")
    print("  • 2020ApJ...904...15K")

    while True:
        bibcode = input("\n📝 Enter the NASA/ADS bibcode (or 'c' to abort): ").strip()

        if bibcode.lower() == "c":
            return None

        if not bibcode:
            print("❌ Bibcode cannot be empty. Please try again.")
            continue

        # Basic validation - should have year and journal format
        if len(bibcode) < 10:
            print("❌ Bibcode seems too short. Please check and try again.")
            continue

        # # Confirm with user
        # print(f"\n📋 You entered: {bibcode}")
        # confirm = input("Is this correct? (y/N): ").strip().lower()
        # if confirm == "y":
        return bibcode
        # else:
        #     print("Let's try again...")


def run_script(script_choice):
    """Run the specified script as a module and handle errors."""
    # Map script choices to module functions
    module_map = {
        "A": A_get_new_DB,
        "B": B_update_UCC_DB,
        "C": C_process_member_files,
        "D": D_update_UCC_site,
    }

    module = module_map.get(script_choice)
    if not module:
        print(f"\n❌ Error: Invalid script choice {script_choice}")
        return False

    script_name = f"{script_choice}_script (imported as module)"
    print(f"\n🚀 Running {script_name}...")
    print("-" * 40)

    # Special handling for script A - requires ADS_bibcode parameter
    if script_choice == "A":
        ads_bibcode = get_ads_bibcode()
        if not ads_bibcode:
            print("❌ Operation cancelled - ADS bibcode is required")
            return False
        module.main(ads_bibcode)
    else:
        # Run the module's main function
        module.main()

    print("-" * 40)
    print(f"✅ {script_name} completed successfully!")
    return True



def get_user_choice():
    """Get and validate user input."""
    while True:
        choice = input("\nEnter your choice (A/B/C/D/Q): ").strip().upper()

        if choice in ["A", "B", "C", "D", "Q"]:
            return choice
        else:
            print("❌ Invalid choice. Please enter A, B, C, D, or Q.")


def main():
    """Main program loop."""
    print("🔬 UCC (Unified Cluster Catalogue) Update System")
    print("\n⚠️  IMPORTANT WORKFLOW SEQUENCE:")
    print("   These scripts must be run in order: A → B → C → D")
    print("   Each script depends on the output of the previous one.")
    print("\n📋 Before running:")
    print("   • Make sure you have the required environment activated")

    # Script mapping for display purposes
    scripts = {
        "A": "A_get_new_DB",
        "B": "B_update_UCC_DB",
        "C": "C_process_member_files",
        "D": "D_update_UCC_site",
    }

    while True:
        display_menu()
        choice = get_user_choice()

        if choice == "Q":
            print("\n👋 Goodbye!")
            sys.exit(0)

        elif choice in scripts:
            success = run_script(choice)

            if success:
                # Ask if user wants to continue to next script
                if choice in ["A", "B", "C"]:  # Not for D (last script)
                    next_script = chr(ord(choice) + 1)  # A->B, B->C, C->D
                    continue_prompt = (
                        input(f"\n❓ Run next script ({next_script}) now? (y/N): ")
                        .strip()
                        .lower()
                    )
                    if continue_prompt == "y":
                        run_script(next_script)
            else:
                print(
                    "\n⚠️  Script execution failed. Please check the error messages above."
                )

            input("\nPress Enter to return to menu...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
