nn(neural1,[I1,I2,Carry],O,[0,1,2,3,4,5,6,7,8,9]) :: neural1(I1,I2,Carry,O).
nn(neural2,[I1,I2,Carry],NewCarry,[0,1]) :: neural2(I1,I2,Carry,NewCarry).


slot(I1,I2,Carry,Carry2,O) :-
    neural1(I1,I2,Carry,O),
    neural2(I1,I2,Carry,Carry2).

add([],[],C,C,[]).

add([H1|T1],[H2|T2],C,Carry,[Digit|Res]) :-
    add(T1,T2,C,Carry2,Res),
    slot(H1,H2,Carry2,Carry,Digit).

add(L1,L2,C,[Carry|Res]) :- add(L1,L2,C,Carry,Res).