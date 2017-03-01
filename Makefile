
target := bhc

all: $(addsuffix .html,$(target)) 

FLAGS := -c imgRecSrc.conf

$(addsuffix .html,$(target)) : %.html : %.jemdoc imgRecSrc.conf MENU
	jemdoc $(FLAGS) $<

