--cleaning data 
select*
from [portfolio project]..Nashhousing


---------------------------------------
--STANDERDIZE DATE FORMAT
select new_Saledate, cast(saledate as date)as new_date
from [portfolio project]..Nashhousing

alter table nashhousing
add new_saledate date;

update Nashhousing
set new_SaleDate = convert(date,SaleDate)

--housing addresss mei jo null values thi unko dhund k unme address dala.
select a.ParcelID,a.PropertyAddress,b.ParcelID,b.PropertyAddress,
ISNULL(a.propertyaddress ,b.PropertyAddress) 
from [portfolio project]..Nashhousing a
JOIN [portfolio project].dbo.Nashhousing b
 on a.ParcelID   = b.ParcelID
 and a.[UniqueID ]<> b.[UniqueID ]
where a.PropertyAddress is null

update a
set propertyaddress = ISNULL(a.propertyaddress ,b.PropertyAddress)
from [portfolio project]..Nashhousing a
JOIN [portfolio project].dbo.Nashhousing b
 on a.ParcelID   = b.ParcelID
 and a.[UniqueID ]<> b.[UniqueID ]
where a.PropertyAddress is null


--breaking out address into individual columns(address,city,state)
--there is another method to break things which is paresename.
-- we replced the commas by period(.) and it works in descending order 3,2,1
select 
parsename(replace(owneraddress,',','.'),3),
parsename(replace(owneraddress,',','.'),2),
parsename(replace(owneraddress,',','.'),1)
from [portfolio project].dbo.nashhousing

alter table nashhousing
add ownersplitaddress nvarchar(255);

update Nashhousing
set ownersplitaddress = parsename(replace(owneraddress ,',','.'),3)

alter table nashhousing
add ownersplitcity nvarchar(255);

update nashhousing
set ownersplitcity =  parsename(replace(owneraddress,',','.'),2)

alter table nashhousing
add ownersplitstate nvarchar(255);

update nashhousing
set ownersplitstate =  parsename(replace(owneraddress,',','.'),1)

select *
from [portfolio project]..Nashhousing



--change y and n to Yes and No in "soldasvacant"
--The SQL DISTINCT keyword is used in conjunction with the SELECT statement 
--to eliminate all the duplicate records and fetching only unique records. 
--There may be a situation when you have multiple duplicate records in a table.
select distinct(soldasvacant),count(soldasvacant)
from [portfolio project]..Nashhousing
group by SoldAsVacant
order by 2
--now we will use conditonal statement
select soldasvacant
,case when soldasvacant = 'y'  then 'yes'
     when soldasvacant = 'N' then 'no'
	 else SoldAsVacant 
	 end
from [portfolio project]..Nashhousing

update nashhousing
set soldasvacant = case when soldasvacant = 'y'  then 'yes'
     when soldasvacant = 'N' then 'no'
	 else SoldAsVacant 
	 end

--------------------------------------------------------
--REMOVE DUPLICATES
-- WE WILL USE CTE 
WITH rownumcte as(
select *,
	row_number() over (
	partition by parcelid,
				 propertyaddress,
				 saleprice,
				 legalreference
				 order by 
					uniqueid
					)row_num
from [portfolio project]..Nashhousing
)
delete
from rownumcte
where row_num > 1
--order by propetyaddress

-------------------------------------------------
--REMOVE UNUSED COLUMN

select *
from [portfolio project]..Nashhousing
ALTER TABLE nashhousing
drop column owneraddress,taxdistrict,propertyaddress
