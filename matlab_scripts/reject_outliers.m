function [data] = reject_outliers(data,m)

    d = abs(data - median(data));
    mdev = median(d);
    
    if mdev
        s = d/mdev;
    else
        s = d/1;
    end

    data = data(s<m);
end