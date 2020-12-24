nn(nn_permute, [X],Y,[0,1,2,3,4,5]) :: net1(X,Y).
nn(nn_op1, [X],Y, [plus,minus,times,div]) :: net2(X,Y).
nn(nn_swap, [X],Y, [no_swap,swap]) :: net3(X,Y).
nn(nn_op2, [X],Y, [plus,minus,times,div]) :: net4(X,Y).

permute(0,A,B,C,A,B,C).
permute(1,A,B,C,A,C,B).
permute(2,A,B,C,B,A,C).
permute(3,A,B,C,B,C,A).
permute(4,A,B,C,C,A,B).
permute(5,A,B,C,C,B,A).


swap(no_swap,X,Y,X,Y).
swap(swap,X,Y,Y,X).

operator(plus,X,Y,Z) :- Z is X+Y.
operator(minus,X,Y,Z) :- Z is X-Y.
operator(times,X,Y,Z) :- Z is X*Y.
operator(div,X,Y,Z) :- Y > 0, 0 =:= X mod Y,Z is X//Y.

wap(Repr,X1,X2,X3,Out) :-
    net1(Repr,Perm),
    permute(Perm,X1,X2,X3,N1,N2,N3),
    net2(Repr,Op1),
    operator(Op1,N1,N2,Res1),
    net3(Repr,Swap),
    swap(Swap,Res1,N3,X,Y),
    net4(Repr,Op2),
    operator(Op2,X,Y,Out).
