csv_file="dde.csv"
f=1
k=1
r=000001
zeros1="000"
zeros2="00000"
zeros3="0000"
while IFS=, read -r  ms T rs Z lg P t0 e w i Mp Rp n
do
	python  varsim.py --star_params  $ms $rs $T $lg $Z --planet_params $t0 $P $e $i $w $Rp $Mp -o data/varsource_000"$n".txt --time 750 --star constant
done < "$csv_file"



