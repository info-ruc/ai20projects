nn(coin_net, [X],Y, [heads,tails]) :: neural_coin(X,Y).
nn(colour_net, [R,G,B],Y, [red, green, blue]) :: neural_colour([R,G,B],Y).

t(0.5) :: col(1,red); t(0.5) :: col(1,blue).
t(0.333) :: col(2,red); t(0.333) :: col(2,green); t(0.333) :: col(2,blue).
t(0.5) :: coin_heads.

outcome(heads,red,_,win).
outcome(heads,_,red,win).
outcome(_,C,C,win).
outcome(Coin,Colour1,Colour2,loss) :- \+outcome(Coin,Colour1,Colour2,win).

game(Coin,Urn1,Urn2,Result) :-
    coin(Coin,Side),
    urn(1,Urn1,C1),
    urn(2,Urn2,C2),
    outcome(Side,C1,C2,Result).

urn(ID,Colour,C) :-
    col(ID,C),
    neural_colour(Colour,C).

coin(Coin,heads) :-
    neural_coin(Coin,heads),
    coin_heads.

coin(Coin,tails) :-
    neural_coin(Coin,tails),
    \+coin_heads.
