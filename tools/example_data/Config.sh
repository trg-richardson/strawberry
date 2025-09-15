
# Basic code operation

    LEAN

    PERIODIC
    SELFGRAVITY
    RANDOMIZE_DOMAINCENTER
    
# Gravity options

    PMGRID=256
    TREEPM_NOTIMESPLIT
    ASMTH=2.0
    
# Softening types and particle types

    NSOFTCLASSES=1
    NTYPES=2

# Floating point accuracy

    POSITIONS_IN_32BIT
    DOUBLEPRECISION=2

# Group finding

    FOF
    SUBFIND
#    MERGERTREE

# Miscellaneous code options
    EVALPOTENTIAL
    POWERSPEC_ON_OUTPUT
    OUTPUT_POTENTIAL
    OUTPUT_ACCELERATION

# IC generation via N-GenIC

    NGENIC=256
    NGENIC_2LPT
    CREATE_GRID
    IDS_32BIT

