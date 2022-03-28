select*
From [portfolio project]..coviddeaths
order by 3,4
select*
From [portfolio project]..[covidvaccinations']

--select data that we are going to use
select location,date,total_cases, new_cases,total_deaths,population
From [portfolio project]..coviddeaths
order by 1,2

--Looking at toal cases vs total deaths
select location,date,total_cases,total_deaths,(total_deaths/total_cases)*100 as Death_Percentage
From [portfolio project]..coviddeaths
where location like 'india'
order by 1,2

-- looking at the total cases vs popultion
--shows what percentage of pop got covid
select location,date,total_cases,population,(total_cases/population)*100 as cases_Percentage
From [portfolio project]..coviddeaths
where location like 'andorra'
order by 1,2

--countries with highest infection rate compared to population
select location,population,max(total_cases) as highest_infected, max((total_cases/population))*100 as population_infected_Percentage
From [portfolio project]..coviddeaths
group by location,population
order by population_infected_Percentage desc

--countries with highest death count 
--cast function is used  to change the data type
select location, population, max(cast(total_deaths as int))as highest_deaths, max((total_deaths/population))*100 as percentage_deaths
from [portfolio project]..coviddeaths
where continent is not null
group by location,population
order by highest_deaths desc

--continents with highest death count per population
select continent, max(cast(total_deaths as int))as highest_Deaths_conti
from [portfolio project]..coviddeaths
where continent is not null
group by continent
order by highest_Deaths_conti desc

-- global numbers 
select sum(new_cases) as total_cases,sum(cast(new_deaths as int))as total_deaths,
 sum(cast(new_deaths as int))/sum(new_cases)*100 as total_death_perctage
from [portfolio project]..coviddeaths
where continent is not null
--group by date
order by 1,2

--vaccination of india
select date,location, cast(total_tests as int)as total_test
from [portfolio project]..[covidvaccinations]
where location like 'india'
order by 1,2


select *
from [portfolio project]..covidvaccinations

--total vaccination vs total population
Select dea.continent,dea.location,dea.date,dea.population,vac.new_vaccinations
,sum(cast(vac.new_vaccinations as bigint)) OVER (partition by dea.location order by dea.location,
 (cast(dea.date as date))) as per_day_new_vaccinations
 -- we will get an error if we use this nwly creTED COLUMN,(per_day_new_vaccinations/population)*100
 -- SO TO AVOID THAT WE WILL DO CTE
from [portfolio project]..coviddeaths dea
join [portfolio project]..[covidvaccinations] vac
	on dea.location = vac.location
	and dea.date = vac.date
where dea.continent is not null
order by 1,2,3

--USE CTE
--now we can use this newly created per_day_vaccination
With PopvsVac (continent , location ,date ,population, new_vaccination ,per_day_new_vaccinations)
as
(
Select dea.continent,dea.location,dea.date,dea.population,vac.new_vaccinations
,sum(cast(vac.new_vaccinations as bigint)) OVER (partition by dea.location order by dea.location,
 (cast(dea.date as date))) as per_day_new_vaccinations
 -- we will get an error if we use this nwly creTED COLUMN,(per_day_new_vaccinations/population)*100
 -- SO TO AVOID THAT WE WILL DO CTE
from [portfolio project]..coviddeaths dea
join [portfolio project]..[covidvaccinations] vac
	on dea.location = vac.location
	and dea.date = vac.date
where dea.continent is not null
--order by 1,2,3
)
Select*,( per_day_new_vaccinations/population)*100
from popvsvac

--TEMP TABLE
drop TABLE if exists #percentpopulationvaccinated
create table #percentpopulationvaccinated
(
continent nvarchar(255),
location nvarchar(255),
date datetime,
population numeric,
new_vaccinations numeric,
per_day_new_vaccinations numeric
)

insert into #percentpopulationvaccinated
Select dea.continent,dea.location,dea.date,dea.population,vac.new_vaccinations
,sum(cast(vac.new_vaccinations as bigint)) OVER (partition by dea.location order by dea.location,
 (cast(dea.date as date))) as per_day_new_vaccinations
 -- we will get an error if we use this nwly creTED COLUMN,(per_day_new_vaccinations/population)*100
 -- SO TO AVOID THAT WE WILL DO CTE
from [portfolio project]..coviddeaths dea
join [portfolio project]..[covidvaccinations] vac
	on dea.location = vac.location
	and dea.date = vac.date
--where dea.continent is not null
--order by 1,2,3

Select*,( per_day_new_vaccinations/population)*100
from #percentpopulationvaccinated

-- CREATING VIEW TO STORE DATA FOR LATER VISUALIZATION

Create view percent_population_vaccinated AS
Select dea.continent,dea.location,dea.date,dea.population,vac.new_vaccinations
,sum(cast(vac.new_vaccinations as bigint)) OVER (partition by dea.location order by dea.location,
 (cast(dea.date as date))) as per_day_new_vaccinations
 -- we will get an error if we use this nwly creTED COLUMN,(per_day_new_vaccinations/population)*100
 -- SO TO AVOID THAT WE WILL DO CTE
from [portfolio project]..coviddeaths dea
join [portfolio project]..[covidvaccinations] vac
	on dea.location = vac.location
	and dea.date = vac.date
where dea.continent is not null
--order by 1,2,3












