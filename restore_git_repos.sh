
# Repository: external/RL-negotiation
echo -e "${BLUE}Restoring: external/RL-negotiation${NC}"
if [ -d "external/RL-negotiation/.git" ]; then
    echo -e "  ${YELLOW}Directory already exists, skipping...${NC}"
else
    # Create parent directory if needed
    mkdir -p "external"
    
    # Clone the repository
    if git clone "git@github.com:yasserfarouk/renting.git" "external/RL-negotiation"; then
        echo -e "  ${GREEN}✓${NC} Successfully cloned"
        
        # Checkout the original branch if not already on it
        cd "external/RL-negotiation"
        current=$(git rev-parse --abbrev-ref HEAD)
        if [ "$current" != "main" ]; then
            if git checkout "main" 2>/dev/null; then
                echo -e "  ${GREEN}✓${NC} Checked out branch: main"
            else
                echo -e "  ${YELLOW}⚠${NC} Could not checkout branch: main"
            fi
        fi
        cd - > /dev/null
    else
        echo -e "  ${RED}✗${NC} Failed to clone"
    fi
fi


# Repository: external/RL-negotiation
echo -e "${BLUE}Restoring: external/RL-negotiation${NC}"
if [ -d "external/RL-negotiation/.git" ]; then
    echo -e "  ${YELLOW}Directory already exists, skipping...${NC}"
else
    # Create parent directory if needed
    mkdir -p "external"
    
    # Clone the repository
    if git clone "git@github.com:yasserfarouk/renting.git" "external/RL-negotiation"; then
        echo -e "  ${GREEN}✓${NC} Successfully cloned"
        
        # Checkout the original branch if not already on it
        cd "external/RL-negotiation"
        current=$(git rev-parse --abbrev-ref HEAD)
        if [ "$current" != "main" ]; then
            if git checkout "main" 2>/dev/null; then
                echo -e "  ${GREEN}✓${NC} Checked out branch: main"
            else
                echo -e "  ${YELLOW}⚠${NC} Could not checkout branch: main"
            fi
        fi
        cd - > /dev/null
    else
        echo -e "  ${RED}✗${NC} Failed to clone"
    fi
fi


# Repository: external/RL-negotiation
echo -e "${BLUE}Restoring: external/RL-negotiation${NC}"
if [ -d "external/RL-negotiation/.git" ]; then
    echo -e "  ${YELLOW}Directory already exists, skipping...${NC}"
else
    # Create parent directory if needed
    mkdir -p "external"
    
    # Clone the repository
    if git clone "git@github.com:yasserfarouk/renting.git" "external/RL-negotiation"; then
        echo -e "  ${GREEN}✓${NC} Successfully cloned"
        
        # Checkout the original branch if not already on it
        cd "external/RL-negotiation"
        current=$(git rev-parse --abbrev-ref HEAD)
        if [ "$current" != "main" ]; then
            if git checkout "main" 2>/dev/null; then
                echo -e "  ${GREEN}✓${NC} Checked out branch: main"
            else
                echo -e "  ${YELLOW}⚠${NC} Could not checkout branch: main"
            fi
        fi
        cd - > /dev/null
    else
        echo -e "  ${RED}✗${NC} Failed to clone"
    fi
fi

