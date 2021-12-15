function [normalized] = normalize_s1(input_image)
    
    d = reject_outliers(input_image(:), 8); 
    normalized = (input_image - min(d))/(max(d) - min(d));

end