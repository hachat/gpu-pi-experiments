#!/bin/bash
echo "RAND()"
for i in {1..14}
do
	#echo "Test $i:"
	./pi_pthread "$i" 0
done

echo "PRNG"
for i in {1..14}
do
	./pi_pthread "$i" 1
done

