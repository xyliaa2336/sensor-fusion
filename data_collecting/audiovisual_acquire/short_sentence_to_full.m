function full_sentence = short_sentence_to_full(short_sentence)
    full_sentence = {};
    commands = {'bin', 'lay', 'place', 'set'};
    colors = {'blue', 'green', 'red', 'white'};
    prepositions = {'at', 'by', 'in', 'with'};
    letters = {'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i', 'h', 'k', 'j', 'm', 'l', 'o', 'n', 'q', 'p', 's', 'r', 'u', 't', 'v', 'y', 'x', 'z'};
    numbers = {'1', '2', '3', '4', '5', '6', '7', '8', '9', 'zero'};
    adverbs = {'again', 'now', 'please', 'soon'};
    cmd_idx    = strncmp(short_sentence(1), commands , 1);
    color_idx  = strncmp(short_sentence(2), colors , 1);
    prep_idx   = strncmp(short_sentence(3), prepositions , 1);
    letter_idx = strncmp(short_sentence(4), letters , 1);
    number_idx = strncmp(short_sentence(5), numbers , 1);
    adverb_idx = strncmp(short_sentence(6), adverbs , 1);
    
    % Put it together
    full_sentence = {commands{cmd_idx}, colors{color_idx}, prepositions{prep_idx}, ...
                     letters{letter_idx}, numbers{number_idx}, adverbs{adverb_idx}};
end