function deb=Debug(level)
    % see npg/pnpg.m for usage

    strlen=0;
    log='';
    add='';
    str='';

    deb.level=@levelGE;
    deb.log=@getLog;
    deb.clearLog=@clearLog;
    deb.print=@print;
    deb.printWithoutDel=@printWithoutDel;
    deb.println=@println;
    deb.clear_print=@clear_print;
    deb.appendLog=@appendLog;
    deb.reset=@reset;

    function res=reset(l)
        strlen=0; log=''; add=''; str='';
    end
    function res=levelGE(l)
        res=(level>=l);
    end
    function res=getLog()
        res=log;
    end
    function clearLog()
        log='';
    end
    function printWithoutDel(l,str)
        if(level>=l)
            add=[add str];
        end
    end
    function print(l,s)
        if(level>=l)
            str=[str s];
        end
    end
    function println(l)
        if(level>=l)
            fprintf('\n');
            strlen=0;
        elseif(l==10)  % for debug to print the size of previous buffer
            fprintf('strlen=%d\n',strlen);
        end
    end
    function ret=clear_print(l)
        ret=false;
        if(level>=l)
            if(isempty(add))
                fprintf([repmat('\b',1,strlen),str,'     ']);
                strlen=length(str)+5;
            else
                fprintf([repmat('\b',1,strlen),str,add,'   \n']);
                add='';
                strlen=0;
                ret=true;
            end
            str='';
        end
    end

    function appendLog(str)
        log=[log str];
    end
end
